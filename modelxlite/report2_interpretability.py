import io
import base64
import math
from typing import Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from sklearn.inspection import PartialDependenceDisplay

from .html_utils import wrap_html, save_html
from .utils import maybe_import_lime, maybe_import_eli5


# -----------------------------
# HTML helpers
# -----------------------------
def _fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;'/>"


def _ensure_df(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(feature_names))


# -----------------------------
# Model type detection
# -----------------------------
def _is_tree_model(model) -> bool:
    """
    Detect common tree/boosting models including XGBoost sklearn wrapper.
    """
    name = model.__class__.__name__.lower()
    mod = model.__class__.__module__.lower()
    hints = [
        "xgb", "xgboost", "randomforest", "gradientboost",
        "xgbclassifier", "xgbregressor",
        "lgbm", "lightgbm",
        "catboost",
        "histgradientboosting",
        "extratrees", "decisiontree"
    ]
    return any(h in name for h in hints) or ("xgboost" in mod) or ("lightgbm" in mod) or ("catboost" in mod)


def _pick_shap_explainer(model, X_background: pd.DataFrame) -> Tuple[Any, str]:
    """
    Returns (explainer, kind) where kind in {"tree","linear","kernel"}.
    """
    # Tree explainer (XGBClassifier etc.)
    if _is_tree_model(model):
        return shap.TreeExplainer(model), "tree"

    # Linear explainer (optional)
    try:
        from sklearn.linear_model import LogisticRegression, LinearRegression
        if isinstance(model, (LogisticRegression, LinearRegression)):
            return shap.LinearExplainer(model, X_background, feature_perturbation="interventional"), "linear"
    except Exception:
        pass

    # Kernel explainer fallback (slow)
    if hasattr(model, "predict_proba"):
        f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X_background.columns))[:, 1]
    else:
        f = lambda data: model.predict(pd.DataFrame(data, columns=X_background.columns))

    bg = shap.sample(X_background, min(200, len(X_background)), random_state=42)
    return shap.KernelExplainer(f, bg), "kernel"


# -----------------------------
# Classification report rows (numeric table)
# -----------------------------
def _numeric_classification_report(y_true, y_hat) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    labels = np.unique(np.concatenate([y_true, y_hat]))

    p, r, f1, s = precision_recall_fscore_support(y_true, y_hat, labels=labels, zero_division=0)
    df = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f1, "support": s})

    acc = float(accuracy_score(y_true, y_hat))
    df2 = pd.DataFrame([{"class": "accuracy", "precision": None, "recall": None, "f1": acc, "support": int(len(y_true))}])
    return pd.concat([df, df2], ignore_index=True)


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    """
    Clean numeric HTML table. (Avoids ugly text blobs.)
    """
    def _fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return float_fmt.format(x)
        return str(x)

    cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join([f"<td>{_fmt(r[c])}</td>" for c in cols]) + "</tr>")

    head = "<tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr>"
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


# -----------------------------
# Main report builder
# -----------------------------
def build_report2_interpretability(
    out_dir,
    cfg,
    model,
    X_test,
    feature_names=None,
    shap_local_index: int = 0,
    pdp_features: Optional[list] = None
):
    """
    Generates:
      02_interpretability.html

    Contains:
      - SHAP Global (TreeExplainer for XGBClassifier)
      - SHAP Local (waterfall)
      - PDP (optional)
      - LIME Local (optional)
      - ELI5 Global + Local (optional)
    """
    sections = []

    X_test = _ensure_df(X_test, feature_names=feature_names)

    # SHAP sample size (if config has it)
    shap_sample = getattr(cfg, "shap_sample", 2000)
    Xs = X_test.sample(min(len(X_test), int(shap_sample)), random_state=42) if len(X_test) else X_test

    # ------------------ SHAP GLOBAL ------------------
    try:
        explainer, kind = _pick_shap_explainer(model, Xs)

        if kind == "tree":
            sv = explainer.shap_values(Xs)

            # Binary classifiers sometimes return [class0, class1]
            if isinstance(sv, list) and len(sv) >= 2:
                sv_plot = sv[1]
            else:
                sv_plot = sv

            plt.figure()
            shap.summary_plot(sv_plot, Xs, show=False)
            sections.append(("SHAP Global Importance", _fig_to_html(plt.gcf())))
        else:
            sv = explainer.shap_values(Xs)
            plt.figure()
            shap.summary_plot(sv, Xs, show=False)
            sections.append(("SHAP Global Importance", _fig_to_html(plt.gcf())))
    except Exception as e:
        sections.append(("SHAP Global Importance", f"<pre class='code'>SHAP global failed: {e}</pre>"))

    # ------------------ SHAP LOCAL ------------------
    try:
        if len(X_test) == 0:
            raise ValueError("X_test is empty")

        idx = int(shap_local_index)
        idx = max(0, min(idx, len(X_test) - 1))
        x_row = X_test.iloc[[idx]]

        # Use a reasonable background set for stability
        Xb = X_test.sample(min(len(X_test), 500), random_state=42)
        explainer, kind = _pick_shap_explainer(model, Xb)

        if kind == "tree":
            sv = explainer.shap_values(x_row)
            base = explainer.expected_value

            # Normalize to class-1 for binary if list
            if isinstance(sv, list) and len(sv) >= 2:
                sv1 = np.asarray(sv[1]).reshape(-1)
                if isinstance(base, (list, np.ndarray)):
                    base1 = float(base[1])
                else:
                    base1 = float(base)
            else:
                sv1 = np.asarray(sv).reshape(-1)
                if isinstance(base, (list, np.ndarray)):
                    base1 = float(base[0])
                else:
                    base1 = float(base)

            exp = shap.Explanation(
                values=sv1,
                base_values=base1,
                data=x_row.iloc[0].values,
                feature_names=x_row.columns.tolist(),
            )
            shap.plots.waterfall(exp, show=False)
            sections.append((f"SHAP Local Explanation (row={idx})", _fig_to_html(plt.gcf())))
        else:
            # Kernel/linear fallback: 1-row summary plot
            sv = explainer.shap_values(x_row)
            plt.figure()
            shap.summary_plot(sv, x_row, show=False)
            sections.append((f"SHAP Local Explanation (row={idx})", _fig_to_html(plt.gcf())))
    except Exception as e:
        sections.append(("SHAP Local Explanation", f"<pre class='code'>SHAP local failed: {e}</pre>"))

    # ------------------ PDP (optional) ------------------
    if pdp_features is not None:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            PartialDependenceDisplay.from_estimator(model, X_test, pdp_features, ax=ax)
            ax.set_title("Partial Dependence Plot(s)")
            sections.append(("Partial Dependence Plot(s)", _fig_to_html(fig)))
        except Exception as e:
            sections.append(("Partial Dependence Plot(s)", f"<pre class='code'>PDP failed: {e}</pre>"))

    # ------------------ LIME (optional) ------------------
    if maybe_import_lime():
        try:
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model has no predict_proba(); LIME classification needs predict_proba().")

            from lime.lime_tabular import LimeTabularExplainer

            idx = int(shap_local_index)
            idx = max(0, min(idx, len(X_test) - 1))

            expl = LimeTabularExplainer(
                training_data=X_test.values,
                feature_names=X_test.columns.tolist(),
                class_names=["0", "1"],
                discretize_continuous=True
            )
            exp = expl.explain_instance(
                X_test.iloc[idx].values,
                model.predict_proba,
                num_features=min(10, X_test.shape[1])
            )
            sections.append((f"LIME Local Explanation (row={idx})", exp.as_html()))
        except Exception as e:
            sections.append(("LIME Local Explanation", f"<pre class='code'>LIME failed: {e}</pre>"))
    else:
        sections.append(("LIME Local Explanation", "<pre class='code'>Optional: pip install lime</pre>"))

    # ------------------ ELI5 (optional) ------------------
    if maybe_import_eli5():
        try:
            import eli5

            # Global weights (works for linear models; may fail for XGB)
            try:
                html_weights = eli5.show_weights(model, feature_names=X_test.columns.tolist()).data
                sections.append(("ELI5 Global (Weights)", html_weights))
            except Exception as e:
                sections.append(("ELI5 Global (Weights)", f"<pre class='code'>ELI5 global weights not available for this model: {e}</pre>"))

            # Local prediction explanation
            try:
                idx = int(shap_local_index)
                idx = max(0, min(idx, len(X_test) - 1))
                html_pred = eli5.show_prediction(model, X_test.iloc[idx]).data
                sections.append((f"ELI5 Local (row={idx})", html_pred))
            except Exception as e:
                sections.append(("ELI5 Local", f"<pre class='code'>ELI5 local failed: {e}</pre>"))
        except Exception as e:
            sections.append(("ELI5", f"<pre class='code'>ELI5 import failed: {e}</pre>"))
    else:
        sections.append(("ELI5", "<pre class='code'>Optional: pip install eli5</pre>"))

    html = wrap_html("02 – Interpretability Report", sections)
    return save_html(out_dir / "02_interpretability.html", html)
