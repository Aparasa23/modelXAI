"""
Enterprise Counterfactuals Engine (DiCE-based)

Goals:
- Provide a consistent interface to generate counterfactuals for classification/regression.
- Attach metadata: deltas, normalized distance, number of features changed, feasibility/actionability flags.
- Support multiple generation methods (random/genetic) and multiple "recourse options" per instance.
- Designed for offline HTML reports: precompute counterfactuals for a selectable set of cases.

This module does NOT change model logic; it wraps DiCE generation and post-processing only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice


@dataclass
class FeatureSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    immutable: bool = False
    # For numeric only
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    # For categorical only
    categories: Optional[List[Any]] = None


@dataclass
class CounterfactualConfig:
    total_cfs: int = 3
    methods: Tuple[str, ...] = ("random", "genetic")  # DiCE supports: random, genetic, kdtree (depending)
    max_cases: int = 25  # precompute how many cases for interactive selector
    seed: int = 7

    # Actionability / feasibility
    immutable_features: Tuple[str, ...] = ()
    permitted_range: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
    # You can express simple monotonic constraints as: {"age": "increase_only", "debt": "decrease_only"}
    direction_constraints: Optional[Dict[str, str]] = None  # "increase_only" | "decrease_only"

    # Distance computation
    distance: str = "normalized_l1"  # "l1" | "normalized_l1"


def infer_feature_specs(
    train_df: pd.DataFrame,
    target_name: str,
    immutable_features: Sequence[str] = (),
    max_unique_for_categorical: int = 25,
) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    imm = set(immutable_features or [])
    for c in train_df.columns:
        if c == target_name:
            continue

        s = train_df[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        if is_num:
            specs.append(
                FeatureSpec(
                    name=c,
                    kind="numeric",
                    immutable=(c in imm),
                    min_val=float(np.nanmin(s.to_numpy(dtype=float))),
                    max_val=float(np.nanmax(s.to_numpy(dtype=float))),
                )
            )
        else:
            # Treat low-cardinality as categorical; otherwise keep as categorical but omit explicit categories.
            cats = None
            nunique = int(s.nunique(dropna=True))
            if nunique <= max_unique_for_categorical:
                cats = [x for x in s.dropna().unique().tolist()]
            specs.append(FeatureSpec(name=c, kind="categorical", immutable=(c in imm), categories=cats))
    return specs


def _continuous_features(specs: Sequence[FeatureSpec]) -> List[str]:
    return [sp.name for sp in specs if sp.kind == "numeric"]


def build_dice_objects(
    train_df: pd.DataFrame,
    target_name: str,
    model: Any,
    problem_type: str,
    feature_specs: Sequence[FeatureSpec],
) -> Tuple[dice_ml.Data, dice_ml.Model]:
    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=_continuous_features(feature_specs),
        outcome_name=target_name,
    )

    if problem_type == "classification":
        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    else:
        m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    return data, m


def _apply_permitted_range(
    query_df: pd.DataFrame,
    permitted_range: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
) -> Dict[str, List[float]]:
    """
    DiCE expects permitted_range as dict of feature -> [min, max] (both required).
    We'll fill missing min/max from query values when unspecified to avoid breaking.
    """
    if not permitted_range:
        return {}

    out: Dict[str, List[float]] = {}
    for feat, (mn, mx) in permitted_range.items():
        v = float(query_df[feat].iloc[0])
        mn2 = v if mn is None else float(mn)
        mx2 = v if mx is None else float(mx)
        if mn2 > mx2:
            mn2, mx2 = mx2, mn2
        out[feat] = [mn2, mx2]
    return out


def _postprocess_counterfactuals(
    query: pd.DataFrame,
    cf_df: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    cfg: CounterfactualConfig,
) -> pd.DataFrame:
    """
    Adds: delta columns, changed flags, l1 distance, normalized distance, feasibility/actionability flags.
    """
    if cf_df is None or len(cf_df) == 0:
        return pd.DataFrame()

    q = query.iloc[0]

    # Compute deltas
    out = cf_df.copy()
    changed = []
    l1 = []
    norm_l1 = []
    feasible = []
    actionable = []

    spec_map = {sp.name: sp for sp in specs}

    for i in range(len(out)):
        row = out.iloc[i]
        n_changed = 0
        l1_sum = 0.0
        norm_sum = 0.0
        is_feasible = True
        is_actionable = True

        for col in query.columns:
            if col not in out.columns:
                continue
            sp = spec_map.get(col)
            if sp is None:
                continue

            a = q[col]
            b = row[col]

            if sp.kind == "numeric":
                try:
                    a_f = float(a)
                    b_f = float(b)
                except Exception:
                    a_f = np.nan
                    b_f = np.nan

                d = b_f - a_f
                out.at[out.index[i], f"Δ {col}"] = d

                if not (pd.isna(a_f) or pd.isna(b_f)) and abs(d) > 1e-12:
                    n_changed += 1

                if not (pd.isna(a_f) or pd.isna(b_f)):
                    l1_sum += abs(d)
                    rng = (sp.max_val - sp.min_val) if (sp.max_val is not None and sp.min_val is not None) else None
                    if rng and rng > 0:
                        norm_sum += abs(d) / float(rng)

                # Feasibility bounds
                if cfg.permitted_range and col in cfg.permitted_range:
                    mn, mx = cfg.permitted_range[col]
                    if mn is not None and b_f < float(mn) - 1e-12:
                        is_feasible = False
                    if mx is not None and b_f > float(mx) + 1e-12:
                        is_feasible = False

                # Direction constraints
                if cfg.direction_constraints and col in cfg.direction_constraints:
                    rule = cfg.direction_constraints[col]
                    if rule == "increase_only" and (not pd.isna(d)) and d < -1e-12:
                        is_feasible = False
                    if rule == "decrease_only" and (not pd.isna(d)) and d > 1e-12:
                        is_feasible = False

                # Actionability: immutable must not change
                if sp.immutable and (not pd.isna(d)) and abs(d) > 1e-12:
                    is_actionable = False

            else:
                # Categorical
                out.at[out.index[i], f"Δ {col}"] = "" if (a == b) else f"{a} → {b}"
                if a != b:
                    n_changed += 1
                if sp.immutable and a != b:
                    is_actionable = False

        changed.append(n_changed)
        l1.append(l1_sum)
        norm_l1.append(norm_sum)
        feasible.append(bool(is_feasible))
        actionable.append(bool(is_actionable))

    out.insert(0, "features_changed", changed)
    out.insert(1, "l1_distance", l1)
    out.insert(2, "normalized_l1_distance", norm_l1)
    out.insert(3, "feasible", feasible)
    out.insert(4, "actionable", actionable)

    # Sort by actionability/feasibility + distance
    if cfg.distance == "normalized_l1":
        out = out.sort_values(by=["actionable", "feasible", "normalized_l1_distance", "features_changed"], ascending=[False, False, True, True])
    else:
        out = out.sort_values(by=["actionable", "feasible", "l1_distance", "features_changed"], ascending=[False, False, True, True])

    return out.reset_index(drop=True)


def generate_counterfactuals_for_instance(
    cfg: CounterfactualConfig,
    dice_data: dice_ml.Data,
    dice_model: dice_ml.Model,
    query: pd.DataFrame,
    problem_type: str,
    feature_specs: Sequence[FeatureSpec],
    desired_class: str = "opposite",
    desired_range: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": { ... },  # dict
        "methods": {
            "random": {"cfs": [..rows..], "table": <pd.DataFrame as dict records>, "meta": {...}},
            ...
        }
      }
    """
    rng = np.random.RandomState(cfg.seed)
    np.random.seed(cfg.seed)

    permitted_range = _apply_permitted_range(query, cfg.permitted_range)

    result: Dict[str, Any] = {
        "query": query.iloc[0].to_dict(),
        "methods": {},
    }

    for method in cfg.methods:
        try:
            dice = Dice(dice_data, dice_model, method=method)

            # DiCE kwargs
            kwargs: Dict[str, Any] = {
                "query_instances": query,
                "total_CFs": int(cfg.total_cfs),
            }
            if permitted_range:
                kwargs["permitted_range"] = permitted_range

            if problem_type == "classification":
                kwargs["desired_class"] = desired_class
            else:
                kwargs["desired_range"] = desired_range

            cf = dice.generate_counterfactuals(**kwargs)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            pp = _postprocess_counterfactuals(query, cf_df, feature_specs, cfg)

            meta = {
                "method": method,
                "n": int(len(pp)),
                "best_actionable": bool(len(pp) > 0 and bool(pp.loc[0, "actionable"])),
                "best_feasible": bool(len(pp) > 0 and bool(pp.loc[0, "feasible"])),
                "best_distance": float(pp.loc[0, "normalized_l1_distance"] if (cfg.distance == "normalized_l1") else pp.loc[0, "l1_distance"]) if len(pp) > 0 else None,
            }

            result["methods"][method] = {
                "table": pp.to_dict(orient="records"),
                "meta": meta,
            }
        except Exception as e:
            result["methods"][method] = {
                "table": [],
                "meta": {"method": method, "error": str(e)},
            }

    return result


def predict_for_query(model: Any, query_df: pd.DataFrame, problem_type: str) -> Dict[str, Any]:
    """
    Adds model prediction context for the original query.
    """
    if problem_type == "classification":
        pred = int(model.predict(query_df)[0])
        proba = None
        try:
            proba = model.predict_proba(query_df)[0].tolist()
        except Exception:
            proba = None
        return {"pred": pred, "proba": proba}
    else:
        pred = float(model.predict(query_df)[0])
        return {"pred": pred}


def precompute_cases(
    cfg: CounterfactualConfig,
    model: Any,
    train_df: pd.DataFrame,
    target_name: str,
    X: pd.DataFrame,
    problem_type: str,
    query_indices: Sequence[int],
) -> Dict[str, Any]:
    specs = infer_feature_specs(train_df, target_name, immutable_features=cfg.immutable_features)
    dice_data, dice_model = build_dice_objects(train_df, target_name, model, problem_type, specs)

    cases = []
    for idx in query_indices:
        idx2 = int(max(0, min(int(idx), len(X) - 1)))
        q = X.iloc[[idx2]].copy()
        pred_ctx = predict_for_query(model, q, problem_type)

        # Regression desired_range: +-10%
        desired_range = None
        if problem_type != "classification":
            y0 = float(pred_ctx["pred"])
            desired_range = [y0 * 0.9, y0 * 1.1]

        cf_payload = generate_counterfactuals_for_instance(
            cfg=cfg,
            dice_data=dice_data,
            dice_model=dice_model,
            query=q,
            problem_type=problem_type,
            feature_specs=specs,
            desired_class="opposite",
            desired_range=desired_range,
        )

        cases.append({
            "index": idx2,
            "pred_ctx": pred_ctx,
            "cf": cf_payload,
        })

    return {
        "feature_specs": [sp.__dict__ for sp in specs],
        "cases": cases,
        "config": cfg.__dict__,
    }
