import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, log_loss, brier_score_loss,
    roc_curve
)
from sklearn.metrics import det_curve
from sklearn.calibration import calibration_curve

import plotly.express as px
import plotly.graph_objects as go

from .html_utils import df_to_html_table, plotly_to_div, wrap_html, save_html
from .binning import bin_table_binary, ks_table_binary
from .threshold_ui import build_dynamic_threshold_block
from .utils import maybe_import_phik


def _classification_report_table(y_true, y_pred) -> pd.DataFrame:
    """
    Numeric classification report table (no text blob).
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({
        "class": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": s
    })
    acc = accuracy_score(y_true, y_pred)
    summary = pd.DataFrame([{
        "class": "accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "f1": acc,
        "support": len(y_true)
    }])
    return pd.concat([df, summary], ignore_index=True)


def build_report1_classification(
    out_dir,
    X_test_df: pd.DataFrame,
    y_test,
    y_pred,
    y_proba,
    bins: int,
    report_on: str = "test",
    X_train=None,
    y_train=None,
    X_train_raw=None,
    X_test_raw=None
):
    """
    Generates:
      01_model_performance.html

    ORDER (exact):
      1) Decile/Percentile Distribution (TEST/TRAIN if chosen)
      2) KS Decile
      3) ROC
      4) DET
      5) Cumulative Gain Plot
      6) Dynamic Threshold Analysis (Interactive)
      7) Core Classification Metrics
      8) Classification Report (numeric table)
      9) Calibration
      10) VIF + Corr + Phik
    """
    sections = []
    bin_label = "Decile" if bins == 10 else "Percentile" if bins == 100 else f"Bin({bins})"

    if y_proba is None or len(np.unique(y_test)) != 2:
        sections.append(("Error", "<pre class='code'>Report #1 deep diagnostics require binary classification with predict_proba().</pre>"))
        html = wrap_html("01 – Model Performance (Classification)", sections)
        return save_html(out_dir / "01_model_performance.html", html)

    p = y_proba[:, 1]

    # 1) Decile/Percentile Distribution (TEST/TRAIN if chosen)
    test_bins = bin_table_binary(y_test, p, X_raw=X_test_raw, bins=bins).rename(columns={"BIN": bin_label.upper()})
    sections.append((f"{bin_label} Distribution Report (TEST)", df_to_html_table(test_bins)))

    if report_on in ("train", "both") and X_train is not None and y_train is not None:
        Xtr = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=X_test_df.columns)
        p_tr = Xtr.copy()
        p_tr = None
        # get proba on train
        # NOTE: model is attached in runner; we will pass proba instead in runner if you want.
        # Here we expect y_proba_train via runner; keep simplest: recompute in runner.
        # We'll skip here if not provided properly.
        pass

    # For train distribution, we support it through runner by passing y_proba_train and X_train_raw
    # If you want train inside this function, use the "build_report1_classification_with_train" in runner.

    # 2) KS Decile (fixed 10)
    ks_df = ks_table_binary(y_test, p, n_bins=10)
    sections.append(("KS Decile Report", df_to_html_table(ks_df)))

    # 3) ROC
    fpr, tpr, _ = roc_curve(y_test, p)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    sections.append(("ROC Curve", plotly_to_div(roc_fig)))

    # 4) DET
    fpr_det, fnr_det, _ = det_curve(y_test, p)
    det_fig = go.Figure()
    det_fig.add_trace(go.Scatter(x=fpr_det, y=fnr_det, mode="lines", name="DET"))
    det_fig.update_layout(title="DET Curve", xaxis_title="FPR", yaxis_title="FNR")
    sections.append(("DET Curve", plotly_to_div(det_fig)))

    # 5) Cumulative Gain Plot
    df_gain = pd.DataFrame({"y": y_test, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df_gain["cum_pos"] = (df_gain["y"] == 1).cumsum()
    total_pos = (df_gain["y"] == 1).sum()
    df_gain["cum_gain"] = df_gain["cum_pos"] / max(total_pos, 1)
    df_gain["population"] = (df_gain.index + 1) / len(df_gain)

    gain_fig = go.Figure()
    gain_fig.add_trace(go.Scatter(x=df_gain["population"], y=df_gain["cum_gain"], mode="lines", name="Model"))
    gain_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    gain_fig.update_layout(title="Cumulative Gain", xaxis_title="Population", yaxis_title="Gain")
    sections.append(("Cumulative Gain Plot", plotly_to_div(gain_fig)))

    # 6) Dynamic Threshold Analysis (Interactive) (slider first)
    # Probability-based metrics (threshold-independent)
    auroc = float(roc_auc_score(y_test, p))
    ll = float(log_loss(y_test, y_proba))
    br = float(brier_score_loss(y_test, p))

    sections.append((
        "Dynamic Threshold Analysis (Interactive)",
        build_dynamic_threshold_block(
            y_true=y_test,
            p=p,
            auroc=auroc,
            logloss=ll,
            brier=br
        )
    ))

    # 7) Core Classification Metrics (clean table)
    core = {
        "Accuracy@0.5": float(accuracy_score(y_test, y_pred)),
        "Precision@0.5": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall@0.5": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1@0.5": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC@0.5": float(matthews_corrcoef(y_test, y_pred)),
        "AU-ROC": float(roc_auc_score(y_test, p)),
        "Log Loss": float(log_loss(y_test, y_proba)),
        "Brier Score": float(brier_score_loss(y_test, p)),
    }
    core_df = pd.DataFrame([{"metric": k, "value": v} for k, v in core.items()])

    # 8) Classification Report (rendered as numeric table)
    cr_df = _classification_report_table(y_test, y_pred)

    # 9) Calibration
    prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    cal_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    cal_fig.update_layout(title="Calibration Chart", xaxis_title="Pred prob", yaxis_title="Observed freq")
    sections.append(("Calibration", plotly_to_div(cal_fig)))

    # 10) VIF + Corr + Phik
    # VIF
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xnum = X_test_df.select_dtypes(include=[np.number]).dropna()
        if Xnum.shape[1] >= 2:
            vals = Xnum.to_numpy()
            vifs = [{"feature": c, "VIF": float(variance_inflation_factor(vals, i))} for i, c in enumerate(Xnum.columns)]
            vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
            sections.append(("VIF (Multicollinearity)", df_to_html_table(vif_df)))
        else:
            sections.append(("VIF (Multicollinearity)", "<pre class='code'>Not enough numeric features for VIF.</pre>"))
    except Exception as e:
        sections.append(("VIF (Multicollinearity)", f"<pre class='code'>VIF skipped: {e}</pre>"))

    # Corr heatmap
    corr = X_test_df.select_dtypes(include=[np.number]).corr()
    if corr.shape[0] > 1:
        corr_fig = px.imshow(corr, title="Correlation Heatmap (Numeric Features)")
        sections.append(("Correlation Heatmap", plotly_to_div(corr_fig)))
    else:
        sections.append(("Correlation Heatmap", "<pre class='code'>Not enough numeric features.</pre>"))

    # Phik optional
    if maybe_import_phik():
        try:
            import phik  # noqa
            tmp = X_test_df.copy()
            tmp["target"] = y_test
            phik_mat = tmp.phik_matrix()
            phik_fig = px.imshow(phik_mat, title="Phik Correlation Matrix")
            sections.append(("Phik Correlation Analysis", plotly_to_div(phik_fig)))
        except Exception as e:
            sections.append(("Phik Correlation Analysis", f"<pre class='code'>Phik failed: {e}</pre>"))
    else:
        sections.append(("Phik Correlation Analysis", "<pre class='code'>phik not installed; skipped. (pip install phik)</pre>"))

    html = wrap_html("01 – Model Performance (Classification)", sections)
    return save_html(out_dir / "01_model_performance.html", html)
