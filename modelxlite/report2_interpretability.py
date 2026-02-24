import io
from pathlib import Path
import base64
import json
import math
from dataclasses import asdict
from typing import Optional, Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap

from sklearn.inspection import PartialDependenceDisplay


from .utils import maybe_import_lime, maybe_import_eli5
from .html_utils import wrap_html, save_html
# Compatibility: html_utils.save_html signature may be (html, out_path) or (out_path, html)
def _save_html_compat(out_path, html):
    """Save HTML and return a Path (works with either save_html signature)."""
    try:
        res = save_html(out_path, html)
    except TypeError:
        res = save_html(html, out_path)
    return Path(res) if not isinstance(res, Path) else res



def _with_purpose(purpose: str, html_block: str) -> str:
    """Wrap a section body with a stakeholder-friendly purpose box (idempotent)."""
    html_block = html_block or ""
    if "data-purpose-box=\"1\"" in html_block:
        return html_block
    return (
        "<div data-purpose-box=\"1\" style=\"margin:8px 0 12px 0; padding:10px 12px; background:#f7f7f7; "
        "border:1px solid #e6e6e6; border-radius:12px;\">"
        "<div style=\"font-weight:800; margin-bottom:4px;\">What this section does</div>"
        "<div style=\"color:#555; font-size:13px; line-height:1.45;\">" + purpose + "</div>"
        "</div>" + html_block
    )


def _apply_section_explanations(sections):
    """Add stakeholder explanations to every section exactly once."""
    def norm(s: str) -> str:
        return (s or "").strip().lower()

    # Fuzzy match rules (substring -> explanation)
    rules = []
    rules.append(("executive", "Purpose: A plain-English summary of what the model is doing and the top drivers. How it helps: lets non-technical readers understand the story in 60 seconds. What to watch for: if top drivers are surprising or look like leakage."))
    rules.append(("global", "Purpose: Explain overall feature influence across the whole dataset. How it helps: shows which inputs matter most to the model’s decisions. What to watch for: dominance by a single suspicious feature or unstable importance across methods."))
    rules.append(("shap", "Purpose: SHAP decomposes a prediction into feature contributions. How it helps: shows direction (+/−) and magnitude of each feature’s impact. What to watch for: very large contributions from leaked features or features with missingness patterns."))
    rules.append(("local explanations", "Purpose: Explain individual predictions case-by-case (why this specific row got this score). How it helps: supports audits, customer explanations, and debugging. What to watch for: inconsistent explanations for similar cases."))
    rules.append(("case selector", "Purpose: Lets you pick a case and view its local explanation interactively. How it helps: stakeholders can explore different examples without rerunning code."))
    rules.append(("pdp", "Purpose: Partial Dependence shows how predictions change as one feature moves (averaged over others). How it helps: reveals monotonicity and non-linear effects. What to watch for: jagged shapes (data sparsity) or effects that contradict domain logic."))
    rules.append(("ice", "Purpose: ICE shows per-row response curves (not averaged). How it helps: reveals heterogeneous behavior across segments. What to watch for: wildly different curves (segment effects)."))
    rules.append(("permutation", "Purpose: Measures how much performance drops when a feature is shuffled. How it helps: model-agnostic importance check. What to watch for: correlated features can dilute importance."))
    rules.append(("eli5", "Purpose: Provides readable feature weights/importance for supported models. How it helps: quick sanity check of top drivers. What to watch for: interpretation limits for non-linear models."))

    DEFAULT = (
        "Purpose: Explain what this section shows in practical terms. "
        "How it helps: supports decision-making, debugging, or governance. "
        "What to watch for: unexpected patterns, instability, or mismatch vs domain knowledge."
    )

    out = []
    for title, body in sections:
        t = norm(title)
        chosen = None
        for sub, expl in rules:
            if sub in t:
                chosen = expl
                break
        if chosen is None:
            chosen = DEFAULT
        out.append((title, _with_purpose(chosen, body)))
    return out

def _ensure_df(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(feature_names))


def _fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _img_tag_from_b64(b64: str, max_width: str = "100%") -> str:
    return f"<img src='data:image/png;base64,{b64}' style='max-width:{max_width};height:auto;'/>"


def _safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.6f}", max_rows: int = 200) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return ""
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(float(v))
        return str(v)

    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    cols = list(df2.columns)
    head = "<tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr>"

    rows_html = []
    for _, r in df2.iterrows():
        rows_html.append("<tr>" + "".join([f"<td>{fmt(r[c])}</td>" for c in cols]) + "</tr>")

    return f"<table><thead>{head}</thead><tbody>{''.join(rows_html)}</tbody></table>"


def _safe_json(obj: Any) -> str:
    """
    JSON for embedding into <script>. Ensures strings are safe enough for inline usage.
    """
    s = json.dumps(obj, ensure_ascii=False)
    # Critical: prevent embedded HTML (e.g., LIME) from prematurely closing the surrounding <script> tag.
    # The HTML parser terminates a <script> element on the literal sequence </script>, even if it appears inside JS strings.
    s = s.replace("</script>", "<\\/script>").replace("</SCRIPT>", "<\\/SCRIPT>")
    # Defensive: avoid HTML comment openers inside script blocks (rare, but can break parsing in some contexts)
    s = s.replace("<!--", "<\\!--")
    return s


# ============================================================
# Data validation / cleaning (fixes your SHAP dtype issues)
# ============================================================
def _coerce_numeric_string_series(s: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detects values like "[6.2637365E-1]" and converts to float.
    Returns (converted_series, log_dict).
    """
    log = {"column": s.name, "converted": False, "bad_examples": [], "num_coerced": 0}
    if s.dtype != object:
        return s, log

    # sample a few for detection
    sample = s.dropna().astype(str).head(50).tolist()
    if not sample:
        return s, log

    looks_numericish = False
    for v in sample:
        v2 = v.strip()
        if v2.startswith("[") and v2.endswith("]"):
            looks_numericish = True
            break
        # scientific notation string like "6.2E-1"
        if any(ch.isdigit() for ch in v2) and ("e" in v2.lower() or "." in v2):
            looks_numericish = True

    if not looks_numericish:
        return s, log

    # attempt conversion:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"^\[", "", regex=True).str.replace(r"\]$", "", regex=True)
    converted = pd.to_numeric(s2, errors="coerce")

    coerced = ((~s.isna()) & (converted.isna())).sum()

    if converted.notna().sum() > 0:
        log["converted"] = True
        log["num_coerced"] = int(coerced)
        bad = s[~s.isna() & converted.isna()].astype(str).head(5).tolist()
        log["bad_examples"] = bad
        return converted, log

    return s, log


def validate_and_clean_X(X: pd.DataFrame, max_cardinality_for_cat: int = 80) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans numeric-looking strings and returns an audit log.
    DOES NOT do encoding; assumes X is already model-ready if needed.
    """
    log: Dict[str, Any] = {
        "n_rows": int(len(X)),
        "n_cols": int(X.shape[1]),
        "dtypes": {c: str(X[c].dtype) for c in X.columns},
        "missing_pct": {c: float(X[c].isna().mean()) for c in X.columns},
        "coercions": [],
        "potential_categoricals": [],
    }

    Xc = X.copy()

    # coerce numeric-looking object columns
    for c in Xc.columns:
        if Xc[c].dtype == object:
            new_s, clog = _coerce_numeric_string_series(Xc[c])
            if clog["converted"]:
                Xc[c] = new_s
                log["coercions"].append(clog)

    # identify categorical-like columns (for reporting only)
    for c in Xc.columns:
        nunique = Xc[c].nunique(dropna=True)
        if nunique <= max_cardinality_for_cat and Xc[c].dtype == object:
            log["potential_categoricals"].append({"column": c, "nunique": int(nunique)})

    log["dtypes_after"] = {c: str(Xc[c].dtype) for c in Xc.columns}
    return Xc, log


# ============================================================
# Explainer router
# ============================================================
def _is_tree_model(model) -> bool:
    name = model.__class__.__name__.lower()
    mod = model.__class__.__module__.lower()
    tree_hints = [
        "xgb", "xgboost",
        "randomforest", "extratrees", "decisiontree",
        "gradientboost", "histgradientboost",
        "lgbm", "lightgbm",
        "catboost",
    ]
    return any(h in name for h in tree_hints) or ("xgboost" in mod) or ("lightgbm" in mod) or ("catboost" in mod)


def _is_linear_model(model) -> bool:
    name = model.__class__.__name__.lower()
    linear_hints = ["logisticregression", "linearregression", "ridge", "lasso", "elasticnet", "sgdclassifier", "sgdregressor"]
    return any(h in name for h in linear_hints)


def make_predict_fn(model, problem_type: str, positive_class: int = 1):
    """
    Returns a callable f(X_df)->pred that SHAP can use.
    For binary classification: returns proba[:,positive_class]
    For regression: returns y_pred
    """
    if problem_type == "classification":
        if hasattr(model, "predict_proba"):
            def f(X):
                Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
                proba = model.predict_proba(Xdf)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, positive_class]
                return proba
            return f

        if hasattr(model, "decision_function"):
            def f(X):
                Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
                scores = model.decision_function(Xdf)
                return 1.0 / (1.0 + np.exp(-scores))
            return f

        def f(X):
            Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            return model.predict(Xdf)
        return f

    def f(X):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return model.predict(Xdf)
    return f


def pick_shap_explainer(model, X_background: pd.DataFrame, problem_type: str):
    """
    Chooses a stable explainer:
      - TreeExplainer for tree models
      - LinearExplainer for linear models
      - KernelExplainer otherwise (small background)
    """
    if _is_tree_model(model):
        return shap.TreeExplainer(model), "tree"

    if _is_linear_model(model):
        return shap.LinearExplainer(model, X_background, feature_perturbation="interventional"), "linear"

    f = make_predict_fn(model, problem_type)
    bg = shap.sample(X_background, min(200, len(X_background)), random_state=42)
    return shap.KernelExplainer(f, bg), "kernel"


# ============================================================
# Case selection (local explanations)
# ============================================================
def select_cases_classification(model, X: pd.DataFrame, y_true: Optional[pd.Series], max_cases: int = 8) -> List[Dict[str, Any]]:
    f = make_predict_fn(model, "classification")
    p = np.asarray(f(X))
    y_pred = (p >= 0.5).astype(int)

    cases = []
    n = len(X)
    if n == 0:
        return cases

    idx_hi = int(np.argmax(p))
    idx_lo = int(np.argmin(p))
    cases.append({"name": "Highest score", "index": idx_hi, "score": float(p[idx_hi])})
    cases.append({"name": "Lowest score", "index": idx_lo, "score": float(p[idx_lo])})

    if y_true is not None:
        yt = np.asarray(y_true).astype(int)
        wrong = np.where(y_pred != yt)[0]
        if len(wrong) > 0:
            conf = np.abs(p[wrong] - 0.5)
            top_wrong = wrong[np.argsort(-conf)][: min(3, len(wrong))]
            for k, ix in enumerate(top_wrong, start=1):
                cases.append({
                    "name": f"Misclassified (confident) #{k}",
                    "index": int(ix),
                    "score": float(p[ix]),
                    "y_true": int(yt[ix]),
                    "y_pred": int(y_pred[ix]),
                })

    seen = set()
    out = []
    for c in cases:
        if c["index"] in seen:
            continue
        seen.add(c["index"])
        out.append(c)
        if len(out) >= max_cases:
            break
    return out


def select_cases_regression(model, X: pd.DataFrame, y_true: Optional[pd.Series], max_cases: int = 8) -> List[Dict[str, Any]]:
    yhat = np.asarray(model.predict(X))
    cases = []
    if len(X) == 0:
        return cases

    idx_hi = int(np.argmax(yhat))
    idx_lo = int(np.argmin(yhat))
    cases.append({"name": "Highest prediction", "index": idx_hi, "pred": float(yhat[idx_hi])})
    cases.append({"name": "Lowest prediction", "index": idx_lo, "pred": float(yhat[idx_lo])})

    if y_true is not None:
        yt = np.asarray(y_true).astype(float)
        resid = np.abs(yt - yhat)
        worst = np.argsort(-resid)[: min(3, len(resid))]
        for k, ix in enumerate(worst, start=1):
            cases.append({
                "name": f"Largest residual #{k}",
                "index": int(ix),
                "pred": float(yhat[ix]),
                "y_true": float(yt[ix]),
                "abs_error": float(resid[ix]),
            })

    seen = set()
    out = []
    for c in cases:
        if c["index"] in seen:
            continue
        seen.add(c["index"])
        out.append(c)
        if len(out) >= max_cases:
            break
    return out


# ============================================================
# SHAP plot builders (save as base64 png)
# ============================================================
def shap_global_plots(explainer, explainer_kind: str, model, Xs: pd.DataFrame, problem_type: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Returns (bar_b64, beeswarm_b64, feature_table_df)
    """
    if explainer_kind == "tree":
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list) and len(sv) >= 2:
            sv = sv[1]
        sv = np.asarray(sv)
    else:
        sv = explainer.shap_values(Xs)
        sv = np.asarray(sv)

    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False)
    bar_b64 = _fig_to_base64_png(plt.gcf())

    plt.figure()
    shap.summary_plot(sv, Xs, show=False)
    bee_b64 = _fig_to_base64_png(plt.gcf())

    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_signed = np.mean(sv, axis=0)
    feat_df = pd.DataFrame({
        "feature": Xs.columns,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "direction": np.where(mean_signed >= 0, "push_up", "push_down")
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return bar_b64, bee_b64, feat_df


def shap_local_waterfall(explainer, explainer_kind: str, X_row: pd.DataFrame, positive_class: int = 1) -> str:
    """
    Returns waterfall plot as base64 png.
    """
    if explainer_kind == "tree":
        sv = explainer.shap_values(X_row)
        base = explainer.expected_value

        if isinstance(sv, list) and len(sv) >= 2:
            sv = np.asarray(sv[positive_class]).reshape(-1)
            if isinstance(base, (list, np.ndarray)):
                base = float(base[positive_class])
            else:
                base = float(base)
        else:
            sv = np.asarray(sv).reshape(-1)
            if isinstance(base, (list, np.ndarray)):
                base = float(base[0])
            else:
                base = float(base)

        exp = shap.Explanation(
            values=sv,
            base_values=base,
            data=X_row.iloc[0].values,
            feature_names=X_row.columns.tolist(),
        )
        shap.plots.waterfall(exp, show=False)
        return _fig_to_base64_png(plt.gcf())

    sv = explainer.shap_values(X_row)
    sv = np.asarray(sv).reshape(-1)
    df = pd.DataFrame({"feature": X_row.columns, "shap": sv}).sort_values("shap", key=np.abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"][::-1], df["shap"][::-1])
    ax.set_title("Local feature contributions (approx)")
    return _fig_to_base64_png(fig)


def dependence_plots(model, X: pd.DataFrame, features: List[Union[int, str]], max_plots: int = 8) -> Dict[str, str]:
    """
    Precompute PDP plots for selected features; returns dict {feature_name: b64_png}
    """
    out = {}
    cols = list(X.columns)
    chosen = []
    for f in features:
        if isinstance(f, int) and 0 <= f < len(cols):
            chosen.append(cols[f])
        elif isinstance(f, str) and f in cols:
            chosen.append(f)

    chosen = chosen[:max_plots]
    for feat in chosen:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            PartialDependenceDisplay.from_estimator(model, X, [feat], ax=ax)
            ax.set_title(f"Partial Dependence: {feat}")
            out[feat] = _fig_to_base64_png(fig)
        except Exception:
            continue
    return out


def prediction_distribution_plot(model, X: pd.DataFrame, problem_type: str) -> str:
    if problem_type == "classification":
        f = make_predict_fn(model, "classification")
        p = np.asarray(f(X))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(p, bins=30)
        ax.set_title("Prediction Probability Distribution")
        ax.set_xlabel("P(class=1)")
        ax.set_ylabel("count")
        return _fig_to_base64_png(fig)
    else:
        yhat = np.asarray(model.predict(X))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(yhat, bins=30)
        ax.set_title("Prediction Distribution")
        ax.set_xlabel("y_pred")
        ax.set_ylabel("count")
        return _fig_to_base64_png(fig)


def rank_stability_check(model, explainer, explainer_kind: str, X: pd.DataFrame, problem_type: str, sample_n: int = 1200) -> Tuple[float, pd.DataFrame]:
    """
    Computes two SHAP global rankings on two random samples and returns Spearman correlation of ranks.
    """
    if len(X) < 50:
        return (np.nan, pd.DataFrame())

    X1 = X.sample(min(sample_n, len(X)), random_state=11)
    X2 = X.sample(min(sample_n, len(X)), random_state=23)

    def global_rank(Xs):
        if explainer_kind == "tree":
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list) and len(sv) >= 2:
                sv = sv[1]
            sv = np.asarray(sv)
        else:
            sv = np.asarray(explainer.shap_values(Xs))
        mean_abs = np.mean(np.abs(sv), axis=0)
        return pd.Series(mean_abs, index=Xs.columns).sort_values(ascending=False)

    r1 = global_rank(X1)
    r2 = global_rank(X2)

    common = r1.index.intersection(r2.index)
    a = r1.loc[common].rank(ascending=False)
    b = r2.loc[common].rank(ascending=False)

    corr = np.corrcoef(a.values, b.values)[0, 1]
    table = pd.DataFrame({"feature": common, "rank_sample1": a.values, "rank_sample2": b.values})
    table["rank_diff"] = np.abs(table["rank_sample1"] - table["rank_sample2"])
    table = table.sort_values("rank_diff", ascending=False).head(25).reset_index(drop=True)
    return float(corr), table


# ============================================================
# Main: Report 2 builder
# ============================================================
def build_report2_interpretability(
    out_dir,
    cfg,
    model,
    X_test,
    feature_names=None,
    y_test=None,
    shap_local_index: int = 0,
    pdp_features: Optional[list] = None
):
    """
    Generates: 02_interpretability.html

    Interactivity:
      - local case dropdown: swaps waterfall + row snapshot (+ optional LIME/ELI5 snippets)
      - feature dropdown: swaps PDP image (precomputed)
    """
    problem_type = getattr(cfg, "problem_type", "classification")
    positive_class = getattr(cfg, "positive_class", 1)
    shap_sample = int(getattr(cfg, "shap_sample", 2000))

    sections = []

    # 0) Data validation / cleaning
    X = _ensure_df(X_test, feature_names=feature_names)
    X_clean, audit = validate_and_clean_X(X)

    audit_df = pd.DataFrame([
        {"key": "rows", "value": audit["n_rows"]},
        {"key": "cols", "value": audit["n_cols"]},
        {"key": "coercions", "value": len(audit["coercions"])},
        {"key": "potential_categoricals", "value": len(audit["potential_categoricals"])},
    ])
    audit_html = _df_to_html_table(audit_df, float_fmt="{}")

    coercions_html = "<pre class='code'>" + json.dumps(audit["coercions"], indent=2) + "</pre>" if audit["coercions"] else "<pre class='code'>No numeric-string coercions applied.</pre>"
    sections.append(("Data Preconditions & Cleaning Log", audit_html + "<h4>Coercions</h4>" + coercions_html))

    Xs = X_clean.sample(min(len(X_clean), shap_sample), random_state=42) if len(X_clean) else X_clean

    # 1) Explainer
    try:
        explainer, kind = pick_shap_explainer(model, Xs, problem_type)
        sections.append(("Explainer (Auto)", f"<pre class='code'>Selected SHAP explainer: {kind}</pre>"))
    except Exception as e:
        html = wrap_html("02 – Interpretability Report", [("Error", f"<pre class='code'>Explainer init failed: {e}</pre>")])
        return _save_html_compat(out_dir / "02_interpretability.html", html)

    # 2) Global SHAP
    try:
        bar_b64, bee_b64, feat_table = shap_global_plots(explainer, kind, model, Xs, problem_type)

        miss = X_clean.isna().mean().rename("missing_pct").reset_index().rename(columns={"index": "feature"})
        feat_table = feat_table.merge(miss, on="feature", how="left")
        dominance = float(feat_table["mean_abs_shap"].iloc[0] / max(feat_table["mean_abs_shap"].sum(), 1e-12))

        sections.append(("SHAP Global Importance (Bar)", _img_tag_from_b64(bar_b64)))
        sections.append(("SHAP Global Summary (Beeswarm)", _img_tag_from_b64(bee_b64)))
        sections.append(("Top Features Table", _df_to_html_table(feat_table.head(30))))
        sections.append(("Sanity: Dominance Ratio", f"<pre class='code'>Top1(|SHAP|) / Sum(|SHAP|) = {dominance:.4f} (higher means 1 feature dominates)</pre>"))
    except Exception as e:
        sections.append(("SHAP Global (Failed)", f"<pre class='code'>SHAP global failed: {e}</pre>"))
        feat_table = pd.DataFrame()

    # 3) Prediction distribution
    try:
        pred_b64 = prediction_distribution_plot(model, X_clean, problem_type)
        sections.append(("Prediction Distribution", _img_tag_from_b64(pred_b64)))
    except Exception as e:
        sections.append(("Prediction Distribution", f"<pre class='code'>Failed: {e}</pre>"))

    # 4) Rank stability
    try:
        corr, diff_tbl = rank_stability_check(model, explainer, kind, X_clean, problem_type)
        if not math.isnan(corr):
            msg = f"Spearman rank correlation between SHAP feature ranks on two random samples: {corr:.4f}"
            warn = ""
            if corr < 0.85:
                warn = "\nWARNING: Explanations may be unstable (increase shap_sample / reduce noise / check leakage)."
            sections.append(("Stability Check (Global Rank)", f"<pre class='code'>{msg}{warn}</pre>"))
            if len(diff_tbl):
                sections.append(("Most Unstable Feature Ranks (Top 25)", _df_to_html_table(diff_tbl, float_fmt="{:.0f}")))
    except Exception as e:
        sections.append(("Stability Check", f"<pre class='code'>Skipped: {e}</pre>"))

    # 5) PDP selector (interactive)
    dep_imgs: Dict[str, str] = {}
    try:
        if pdp_features is None:
            if len(feat_table):
                top_feats = feat_table["feature"].head(8).tolist()
                pdp_features = top_feats
            else:
                pdp_features = list(X_clean.columns[:5])

        dep_imgs = dependence_plots(model, X_clean, pdp_features, max_plots=8)
    except Exception:
        dep_imgs = {}

    if dep_imgs:
        dep_keys = list(dep_imgs.keys())
        dep_data = {k: dep_imgs[k] for k in dep_keys}
        dep_html = f"""
        <div>
          <label><b>Select feature</b></label>
          <select id="depSelect">
            {''.join([f"<option value='{k}'>{k}</option>" for k in dep_keys])}
          </select>
          <div style="margin-top:10px;">
            <img id="depImg" style="max-width:100%;height:auto;" />
          </div>
        </div>
        <script>
          (function(){{
          const DEP = { _safe_json(dep_data) };
          const depSel = document.getElementById("depSelect");
          const depImg = document.getElementById("depImg");
          function updateDep() {{
            const k = depSel.value;
            depImg.src = "data:image/png;base64," + DEP[k];
          }}
          depSel.addEventListener("change", updateDep);
          depSel.value = "{dep_keys[0]}";
          updateDep();
          }})();
        </script>
        """
        sections.append(("Feature Sensitivity (PDP) – Interactive Selector", dep_html))
    else:
        sections.append(("Feature Sensitivity (PDP)", "<pre class='code'>PDP plots not available (model/feature constraints).</pre>"))

    # 6) Local explanations (interactive case selector)
    try:
        if problem_type == "classification":
            local_cases = select_cases_classification(model, X_clean, y_test)
        else:
            local_cases = select_cases_regression(model, X_clean, y_test)

        if not local_cases:
            raise ValueError("No cases selected")

        local_payload = {}
        lime_payload = {}
        eli5_payload = {}

        lime_ok = maybe_import_lime()
        eli5_ok = maybe_import_eli5()

        lime_explainer = None
        if lime_ok and problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                from lime.lime_tabular import LimeTabularExplainer
                lime_explainer = LimeTabularExplainer(
                    training_data=X_clean.values,
                    feature_names=X_clean.columns.tolist(),
                    class_names=["0", "1"],
                    discretize_continuous=True
                )
            except Exception:
                lime_explainer = None

        for c in local_cases:
            ix = int(c["index"])
            row = X_clean.iloc[[ix]]

            snap = row.T.reset_index()
            snap.columns = ["feature", "value"]
            snap_html = _df_to_html_table(snap, float_fmt="{}")

            wf_b64 = shap_local_waterfall(explainer, kind, row, positive_class=positive_class)

            local_payload[str(ix)] = {
                "meta": c,
                "waterfall_b64": wf_b64,
                "snapshot_html": snap_html
            }

            if lime_explainer is not None:
                try:
                    exp = lime_explainer.explain_instance(
                        row.iloc[0].values,
                        model.predict_proba,
                        num_features=min(10, row.shape[1])
                    )
                    lime_payload[str(ix)] = exp.as_html()
                except Exception:
                    pass

            if eli5_ok:
                try:
                    import eli5
                    eli5_payload[str(ix)] = eli5.show_prediction(model, row.iloc[0]).data
                except Exception:
                    pass

        options = []
        for c in local_cases:
            ix = int(c["index"])
            label = c["name"]
            if problem_type == "classification":
                if "score" in c:
                    label += f" (p={c['score']:.3f})"
                if "y_true" in c:
                    label += f" [y={c['y_true']} pred={c['y_pred']}]"
            else:
                if "pred" in c:
                    label += f" (pred={c['pred']:.3f})"
                if "abs_error" in c:
                    label += f" [abs_err={c['abs_error']:.3f}]"
            options.append(f"<option value='{ix}'>{label}</option>")

        local_ui = f"""
        <div>
          <label><b>Select case</b></label>
          <select id="caseSelect">
            {''.join(options)}
          </select>

          <div style="margin-top:14px;">
            <h4>Row Snapshot</h4>
            <div id="rowSnap"></div>
          </div>

          <div style="margin-top:14px;">
            <h4>SHAP Waterfall</h4>
            <img id="wfImg" style="max-width:100%;height:auto;" />
          </div>

          <div style="margin-top:14px;">
            <h4>LIME</h4>
            <div id="limeBlock"><pre class="code">Not available / not installed.</pre></div>
          </div>

          <div style="margin-top:14px;">
            <h4>ELI5</h4>
            <div id="eli5Block"><pre class="code">Not available / not installed.</pre></div>
          </div>
        </div>

        <script>
          (function(){{
          const LOCAL = { _safe_json(local_payload) };
          const LIME = { _safe_json(lime_payload) };
          const ELI5 = { _safe_json(eli5_payload) };

          const caseSel = document.getElementById("caseSelect");
          const wf = document.getElementById("wfImg");
          const snap = document.getElementById("rowSnap");
          const lime = document.getElementById("limeBlock");
          const eli5 = document.getElementById("eli5Block");

          function updateCase() {{
            const ix = caseSel.value;
            const obj = LOCAL[ix];

            wf.src = "data:image/png;base64," + obj["waterfall_b64"];
            snap.innerHTML = obj["snapshot_html"];

            if (LIME[ix]) {{
              lime.innerHTML = LIME[ix];
            }} else {{
              lime.innerHTML = "<pre class='code'>Not available / not installed.</pre>";
            }}

            if (ELI5[ix]) {{
              eli5.innerHTML = ELI5[ix];
            }} else {{
              eli5.innerHTML = "<pre class='code'>Not available / not installed.</pre>";
            }}
          }}

          caseSel.addEventListener("change", updateCase);
          updateCase();
          }})();
        </script>
        """
        sections.append(("Local Explanations – Interactive Case Selector", _with_purpose("Pick a single row/case to inspect per-feature contributions. Useful for debugging and stakeholder case-level narratives.", local_ui)))

    except Exception as e:
        sections.append(("Local Explanations", f"<pre class='code'>Local explanation block failed: {e}</pre>"))

    # 7) ELI5 global
    if maybe_import_eli5():
        try:
            import eli5
            w = eli5.show_weights(model, feature_names=X_clean.columns.tolist()).data
            sections.append(("ELI5 Global Weights (HTML)", w))
        except Exception as e:
            sections.append(("ELI5 Global Weights (HTML)", f"<pre class='code'>ELI5 weights not supported for this model: {e}</pre>"))
    else:
        sections.append(("ELI5 Global Weights (HTML)", "<pre class='code'>pip install eli5</pre>"))

    statuses = []
    summary = [
        ("problem_type", getattr(cfg, "problem_type", "")),
        ("model", model.__class__.__name__),
        ("rows_used_for_shap", str(len(Xs))),
        ("features", str(X_clean.shape[1])),
    ]
    sections = _apply_section_explanations(sections)

    sections = _apply_section_explanations(sections)

    html = wrap_html("02 – Interpretability Report", sections, statuses=statuses, summary_items=summary)
    return _save_html_compat(out_dir / "02_interpretability.html", html)