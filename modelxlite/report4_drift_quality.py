import numpy as np
import pandas as pd
import plotly.express as px

from .html_utils import df_to_html_table, plotly_to_div, wrap_html, save_html
from .utils import to_df, psi

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

    html = wrap_html("04 – Drift & Data Quality", sections)
    return save_html(out_dir / "04_drift_quality.html", html)
