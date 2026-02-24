import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px
from .utils import to_df, psi

from .html_utils import df_to_html_table, plotly_to_div, wrap_html, save_html
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
    rules.append(("executive", "Purpose: One-screen health summary of data drift and quality. How it helps: quick go/no-go for deployment or retraining."))
    rules.append(("psi", "Purpose: PSI measures distribution shift between reference (train) and current (test/production). How it helps: detects drift that can degrade model performance. What to watch for: features with high PSI—investigate upstream changes."))
    rules.append(("drift", "Purpose: Identify which features changed and by how much. How it helps: points to root-cause signals and monitoring priorities."))
    rules.append(("missing", "Purpose: Track missingness changes. How it helps: sudden missing spikes often indicate pipeline breaks or schema changes."))
    rules.append(("schema", "Purpose: Verify feature set/types match expectations. How it helps: catches breaking changes early."))
    rules.append(("quality", "Purpose: Basic data sanity checks (ranges, outliers, duplicates). How it helps: prevents silent failures before scoring."))

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

def build_report4_drift_quality(out_dir, X_eval, feature_names, drift_ref=None, drift_cur=None, bins=10):
    X = to_df(X_eval, feature_names=feature_names)
    sections = []

    n_rows = len(X)
    missing = X.isna().mean().sort_values(ascending=False)
    missing_df = pd.DataFrame({"feature": missing.index, "missing_pct": missing.values})
    dup_count = int(X.duplicated().sum())
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    sections.append(("Data Integrity – Summary",
                     f"<pre class='code'>Rows: {n_rows}\nDuplicate rows: {dup_count}\nNumeric features: {len(numeric_cols)}</pre>"))
    sections.append(("Missingness (Top 50)", df_to_html_table(missing_df.head(50))))

    # Outliers (z-score)
    Xnum = X[numeric_cols].dropna()
    if len(numeric_cols) > 0 and len(Xnum) > 20:
        z = (Xnum - Xnum.mean()) / (Xnum.std(ddof=0).replace(0, np.nan))
        outlier_rate = (np.abs(z) > 4).mean().sort_values(ascending=False)
        outlier_df = pd.DataFrame({"feature": outlier_rate.index, "outlier_rate(|z|>4)": outlier_rate.values})
        sections.append(("Outlier Rates (|z|>4)", df_to_html_table(outlier_df.head(50))))
    else:
        sections.append(("Outliers", "<pre class='code'>Not enough numeric data to compute outliers.</pre>"))

    # PSI drift if provided
    if drift_ref is not None and drift_cur is not None:
        Xr = to_df(drift_ref, feature_names=feature_names)
        Xc = to_df(drift_cur, feature_names=feature_names)

        rows = []
        for col in feature_names:
            if pd.api.types.is_numeric_dtype(Xr[col]):
                rows.append({"feature": col, "psi": psi(Xr[col], Xc[col], buckets=min(bins, 10))})
        drift_df = pd.DataFrame(rows).dropna().sort_values("psi", ascending=False)

        sections.append(("PSI Drift Table (Top 200)", df_to_html_table(drift_df.head(200))))

        top = drift_df.head(15)
        if len(top) > 0:
            fig = px.bar(top, x="feature", y="psi", title="Top 15 Features by PSI (Higher = More Drift)")
            sections.append(("PSI Drift Plot", plotly_to_div(fig)))
    else:
        sections.append(("PSI Drift", "<pre class='code'>Skipped. Provide drift_ref and drift_cur.</pre>"))

    sections.append(("Concept Drift (Note)",
                     "<pre class='code'>Concept drift is P(Y|X) changing over time. This lite version provides PSI/data drift + integrity checks.\nAdd DeepChecks integration later for full concept drift, stability, leakage, and train-vs-val diagnostics.</pre>"))
    sections = _apply_section_explanations(sections)

    sections = _apply_section_explanations(sections)

    html = wrap_html("04 – Drift & Data Quality", sections)
    return _save_html_compat(out_dir / "04_drift_quality.html", html)