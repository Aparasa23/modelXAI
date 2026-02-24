import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
from .threshold_ui import build_dynamic_threshold_block
from .utils import maybe_import_phik


# ---------------------------
# Purpose / stakeholder help
# ---------------------------

def _normalize_purpose(p: Union[str, Dict[str, str], None]) -> Dict[str, str]:
    """
    Accept either:
      - dict with keys: purpose, helps, watch
      - string -> treated as "purpose"
      - None -> empty
    """
    if p is None:
        return {"purpose": "", "helps": "", "watch": ""}
    if isinstance(p, dict):
        return {
            "purpose": str(p.get("purpose", "")),
            "helps": str(p.get("helps", "")),
            "watch": str(p.get("watch", "")),
        }
    return {"purpose": str(p), "helps": "", "watch": ""}


def _purpose_box(purpose: Union[str, Dict[str, str], None]) -> str:
    p = _normalize_purpose(purpose)
    # Avoid "one-line" rendering by using <div> blocks and a list.
    return f"""
<div data-purpose-box="1"
     style="margin:12px 0 14px 0;
            padding:14px 16px;
            background:#f7f7f7;
            border:1px solid #e6e6e6;
            border-radius:12px;">
  <div style="font-weight:800; font-size:14px; margin-bottom:8px; color:#111;">
    What this section does
  </div>
  <ul style="margin:0; padding-left:18px; color:#333; font-size:13px; line-height:1.5;">
    <li style="margin:6px 0;"><b>Purpose:</b> {p["purpose"]}</li>
    <li style="margin:6px 0;"><b>How it helps:</b> {p["helps"]}</li>
    <li style="margin:6px 0;"><b>What to watch for:</b> {p["watch"]}</li>
  </ul>
</div>
""".strip()


def _wrap_with_purpose(title: str, body_html: str, purpose_map: Dict[str, Dict[str, str]]) -> str:
    """
    Wrap section with a single purpose box.
    If the body already contains a purpose box marker, don't add another.
    """
    if body_html and 'data-purpose-box="1"' in body_html:
        return body_html
    purpose = purpose_map.get(title)
    if purpose is None:
        purpose = {
            "purpose": "Explain what this output shows in business terms.",
            "helps": "Use it to make go/no-go decisions, compare models, or diagnose issues.",
            "watch": "Look for instability, unexpected patterns, or train/test mismatches."
        }
    return _purpose_box(purpose) + body_html


# ---------------------------
# Decile / percentile table
# ---------------------------

def ks_decile_table(
    y_true,
    p_score,
    bins: int = 10,
    label: str = "DECILE",
) -> pd.DataFrame:
    """
    KS-style table with requested columns:
      DECILE, N, POS, NEG, CUM_POS, CUM_NEG, TPR, FPR, KS,
      P_MIN, P_MAX, P_MEAN, PRECISION, LIFT, RECALL, RESPONDER_%

    Bins ordered from highest p to lowest p (Bin 1 = highest score segment).
    Uses rank-based slicing to avoid qcut issues with tied scores.
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(p_score).astype(float).reshape(-1)

    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if df.empty:
        return pd.DataFrame(columns=[
            label, "N", "POS", "NEG", "CUM_POS", "CUM_NEG",
            "TPR", "FPR", "KS", "P_MIN", "P_MAX", "P_MEAN",
            "PRECISION", "LIFT", "RECALL", "RESPONDER_%"
        ])

    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    total_pos = int((df["y"] == 1).sum())
    total_neg = int((df["y"] == 0).sum())

    # Degenerate
    if total_pos == 0 or total_neg == 0:
        out = pd.DataFrame({
            label: [1],
            "N": [len(df)],
            "POS": [total_pos],
            "NEG": [total_neg],
            "CUM_POS": [total_pos],
            "CUM_NEG": [total_neg],
            "TPR": [np.nan],
            "FPR": [np.nan],
            "KS": [np.nan],
            "P_MIN": [float(df["p"].min())],
            "P_MAX": [float(df["p"].max())],
            "P_MEAN": [float(df["p"].mean())],
            "PRECISION": [total_pos / len(df) if len(df) else np.nan],
            "LIFT": [np.nan],
            "RECALL": [np.nan],
            "RESPONDER_%": [np.nan],
        })
        return out

    n = len(df)
    df[label] = (np.floor(np.arange(n) * bins / n).astype(int) + 1)

    g = df.groupby(label, sort=True)
    out = pd.DataFrame({
        label: g.size().index.astype(int),
        "N": g.size().values.astype(int),
        "POS": g["y"].sum().values.astype(int),
    })
    out["NEG"] = (out["N"] - out["POS"]).astype(int)

    out["P_MIN"] = g["p"].min().values.astype(float)
    out["P_MAX"] = g["p"].max().values.astype(float)
    out["P_MEAN"] = g["p"].mean().values.astype(float)

    out["CUM_POS"] = out["POS"].cumsum().astype(int)
    out["CUM_NEG"] = out["NEG"].cumsum().astype(int)

    out["TPR"] = out["CUM_POS"] / float(total_pos)
    out["FPR"] = out["CUM_NEG"] / float(total_neg)
    out["KS"] = out["TPR"] - out["FPR"]

    overall_pos_rate = float(df["y"].mean()) if len(df) else np.nan
    out["PRECISION"] = out["POS"] / out["N"].replace(0, np.nan)
    out["LIFT"] = out["PRECISION"] / (overall_pos_rate if overall_pos_rate and overall_pos_rate > 0 else np.nan)
    out["RECALL"] = out["TPR"]
    out["RESPONDER_%"] = out["POS"] / float(total_pos)

    cols = [
        label, "N", "POS", "NEG", "CUM_POS", "CUM_NEG",
        "TPR", "FPR", "KS", "P_MIN", "P_MAX", "P_MEAN",
        "PRECISION", "LIFT", "RECALL", "RESPONDER_%"
    ]
    out = out[cols]

    # Formatting
    for c in ["TPR", "FPR", "KS", "P_MIN", "P_MAX", "P_MEAN", "PRECISION", "LIFT", "RECALL", "RESPONDER_%"]:
        out[c] = out[c].astype(float).round(6)
    return out


def _safe_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</script>", "<\/script>").replace("</SCRIPT>", "<\/SCRIPT>")


def _distribution_interactive_block(payload: dict) -> str:
    js_payload = _safe_json(payload)
    return f"""
<div class="dist-card">
  <div class="dist-title">KS / Decile Distribution (Interactive)</div>

  <div class="dist-controls">
    <div class="dist-control">
      <div class="muted tiny">Dataset</div>
      <select id="distDataset" class="dist-select">
        <option value="test">Test</option>
        <option value="train">Train</option>
      </select>
    </div>

    <div class="dist-control">
      <div class="muted tiny">Bins</div>
      <select id="distBins" class="dist-select">
        <option value="10">10 (Deciles)</option>
        <option value="25">25</option>
        <option value="50">50</option>
        <option value="100">100 (Percentiles)</option>
      </select>
    </div>

    <div class="dist-control grow">
      <div class="muted tiny">Quick read</div>
      <div class="muted">
        Bin 1 is highest score segment. Compare <b>Precision</b> and <b>Lift</b> in top bins, and peak <b>KS</b>.
      </div>
    </div>
  </div>

  <div id="distNote" class="muted tiny" style="margin: 6px 0 10px 0;"></div>
  <div id="distTable"></div>
</div>

<script>
const DIST_PAYLOAD = {js_payload};

function esc(x) {{
  if (x === null || x === undefined) return "";
  return String(x).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}}

function renderTable(records) {{
  if (!records || records.length === 0) {{
    return "<pre class='code'>No data available for this selection.</pre>";
  }}
  const cols = Object.keys(records[0]);
  let html = "<div class='tablewrap'><table><thead><tr>";
  for (const c of cols) html += `<th>${{esc(c)}}</th>`;
  html += "</tr></thead><tbody>";
  for (const r of records) {{
    html += "<tr>";
    for (const c of cols) {{
      let v = r[c];
      if (typeof v === "number") {{
        v = Number.isInteger(v) ? v.toString() : v.toFixed(6);
      }}
      html += `<td>${{esc(v)}}</td>`;
    }}
    html += "</tr>";
  }}
  html += "</tbody></table></div>";
  return html;
}}

function update() {{
  const ds = document.getElementById("distDataset").value;
  const bins = document.getElementById("distBins").value;
  const key = ds + "_" + bins;

  const block = DIST_PAYLOAD[key];
  const note = document.getElementById("distNote");

  if (!block) {{
    note.textContent = "Not available for this selection.";
    document.getElementById("distTable").innerHTML = "<pre class='code'>Not available.</pre>";
    return;
  }}

  if (block.actual_bins && String(block.actual_bins) !== String(block.requested_bins)) {{
    note.textContent = `Requested ${{block.requested_bins}} bins; actual bins = ${{block.actual_bins}}.`;
  }} else {{
    note.textContent = "";
  }}

  document.getElementById("distTable").innerHTML = renderTable(block.table);
}}

document.getElementById("distDataset").addEventListener("change", update);
document.getElementById("distBins").addEventListener("change", update);

// Disable TRAIN option if no train payload
if (!DIST_PAYLOAD["train_10"]) {{
  const ds = document.getElementById("distDataset");
  for (const opt of Array.from(ds.options)) {{
    if (opt.value === "train") opt.disabled = true;
  }}
  ds.value = "test";
}}
document.getElementById("distBins").value = "10";
update();
</script>

<style>
.dist-card {{
  background:#fff;
  border: 1px solid #e6e6e6;
  border-radius: 14px;
  padding: 14px;
}}
.dist-title {{
  font-weight: 900;
  margin-bottom: 8px;
}}
.dist-controls {{
  display:flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  align-items: flex-end;
  margin: 8px 0 6px 0;
}}
.dist-control {{ min-width: 220px; }}
.dist-control.grow {{ flex: 1 1 360px; }}
.dist-select {{
  width: 100%;
  padding: 10px 10px;
  border-radius: 10px;
  border: 1px solid #dcdcdc;
  background: #fff;
  outline: none;
}}
.tablewrap {{
  overflow:auto;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
  max-height: 520px;
}}
table {{
  width:100%;
  border-collapse: collapse;
  font-size: 13px;
}}
thead th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background:#f1f1f1;
}}
th, td {{
  border:1px solid #e6e6e6;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}}
tbody tr:nth-child(even) {{ background:#fafafa; }}
.muted {{ color:#666; }}
.tiny {{ font-size: 12px; }}
pre.code {{
  background:#f3f3f3;
  padding: 10px;
  border-radius: 10px;
  overflow-x:auto;
}}
</style>
""".strip()


def _classification_report_table(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f1, "support": s})
    acc = accuracy_score(y_true, y_pred)
    summary = pd.DataFrame([{"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": acc, "support": len(y_true)}])
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
    y_proba_train=None,
    X_train_raw=None,
    X_test_raw=None
):
    """
    01_model_performance.html
    """
    out_dir = Path(out_dir)
    sections = []

    if y_proba is None or len(np.unique(y_test)) != 2:
        sections.append(("Error", "<pre class='code'>Report #1 requires binary classification with predict_proba().</pre>"))
        html = wrap_html("01 – Model Performance (Classification)", sections)
        return save_html(out_dir / "01_model_performance.html", html)

    p = np.asarray(y_proba)[:, 1]

    PURPOSE = {
        "KS / Decile Distribution (Interactive)": {
            "purpose": "Breaks predictions into bins (10/25/50/100) to validate ranking power and targeting efficiency.",
            "helps": "Answer: “If we act on top X% scores, what precision/lift do we get and how many responders do we capture?”",
            "watch": "If Train has much higher lift/KS than Test, you may have overfitting or drift."
        },
        "ROC Curve": {
            "purpose": "Shows the trade-off between true positives and false positives across all thresholds.",
            "helps": "Lets you compare models without choosing a cutoff (AUROC summarizes ranking quality).",
            "watch": "Curve close to diagonal implies weak ranking power (AUROC near 0.5)."
        },
        "DET Curve": {
            "purpose": "Visualizes false positives vs false negatives across thresholds (error trade-off view).",
            "helps": "Useful when costs of misses vs false alarms are asymmetric.",
            "watch": "Large error rates even at best region indicate limited separability."
        },
        "Cumulative Gain Plot": {
            "purpose": "Shows how fast responders are captured when sorting by model score (top‑K targeting).",
            "helps": "Converts scores into an action plan: contact top X% and estimate captured responders.",
            "watch": "If curve is near the random line, the model adds little business value."
        },
        "Dynamic Threshold Analysis (Interactive)": {
            "purpose": "Helps select an operating threshold by showing precision/recall/F1 and other metrics vs cutoff.",
            "helps": "Aligns the model decision boundary with capacity, risk tolerance, and business costs.",
            "watch": "A threshold that looks good on Test may not hold if calibration/drift is poor."
        },
        "Core Classification Metrics": {
            "purpose": "Compact scoreboard of key metrics (threshold-based + threshold-free).",
            "helps": "Quick model comparison and sanity checks.",
            "watch": "High AUROC but low F1@0.5 may imply threshold needs tuning."
        },
        "Classification Report": {
            "purpose": "Per-class precision/recall/F1 and support.",
            "helps": "Shows which class is suffering (often minority class).",
            "watch": "Very low recall on positives means you miss most true responders."
        },
        "Calibration": {
            "purpose": "Checks whether predicted probabilities match observed frequencies.",
            "helps": "Critical when stakeholders interpret probabilities as risk/propensity.",
            "watch": "Systematic over/under-confidence suggests probability calibration is needed."
        },
        "VIF (Multicollinearity)": {
            "purpose": "Detects highly collinear numeric features (redundancy).",
            "helps": "Improves stability and interpretability; reduces unstable coefficients in linear models.",
            "watch": "Very high VIFs (e.g., >10) indicate redundancy."
        },
        "Correlation Heatmap": {
            "purpose": "Shows pairwise linear correlations among numeric features.",
            "helps": "Finds redundancy/leakage candidates quickly.",
            "watch": "Near-perfect correlations suggest duplicated signals."
        },
        "Phik Correlation Analysis": {
            "purpose": "Measures non-linear dependencies (handles mixed types better than Pearson).",
            "helps": "Catches relationships Pearson misses.",
            "watch": "Strong correlations between target and leakage features are a red flag."
        },
    }

    # 1) Interactive distribution payload (TEST/TRAIN x bins)
    dist_payload = {}
    dist_bins_choices = [10, 25, 50, 100]
    for b in dist_bins_choices:
        tbl = ks_decile_table(y_test, p, bins=b, label="DECILE")
        dist_payload[f"test_{b}"] = {
            "requested_bins": b,
            "actual_bins": int(tbl["DECILE"].nunique()) if not tbl.empty else 0,
            "table": tbl.to_dict(orient="records"),
        }

    if report_on in ("train", "both") and y_train is not None and y_proba_train is not None:
        ytr = np.asarray(y_train).reshape(-1)
        ypt = np.asarray(y_proba_train)
        ptr = ypt[:, 1] if (ypt.ndim == 2 and ypt.shape[1] >= 2) else ypt.reshape(-1)
        for b in dist_bins_choices:
            tbl = ks_decile_table(ytr, ptr, bins=b, label="DECILE")
            dist_payload[f"train_{b}"] = {
                "requested_bins": b,
                "actual_bins": int(tbl["DECILE"].nunique()) if not tbl.empty else 0,
                "table": tbl.to_dict(orient="records"),
            }

    body = _distribution_interactive_block(dist_payload)
    sections.append(("KS / Decile Distribution (Interactive)", _wrap_with_purpose("KS / Decile Distribution (Interactive)", body, PURPOSE)))

    # 2) ROC
    fpr, tpr, _ = roc_curve(np.asarray(y_test).reshape(-1), p)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    sections.append(("ROC Curve", _wrap_with_purpose("ROC Curve", plotly_to_div(roc_fig), PURPOSE)))

    # 3) DET
    fpr_det, fnr_det, _ = det_curve(np.asarray(y_test).reshape(-1), p)
    det_fig = go.Figure()
    det_fig.add_trace(go.Scatter(x=fpr_det, y=fnr_det, mode="lines", name="DET"))
    det_fig.update_layout(title="DET Curve", xaxis_title="FPR", yaxis_title="FNR")
    sections.append(("DET Curve", _wrap_with_purpose("DET Curve", plotly_to_div(det_fig), PURPOSE)))

    # 4) Cumulative Gain
    df_gain = pd.DataFrame({"y": np.asarray(y_test).reshape(-1), "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df_gain["cum_pos"] = (df_gain["y"] == 1).cumsum()
    total_pos = int((df_gain["y"] == 1).sum())
    df_gain["cum_gain"] = df_gain["cum_pos"] / max(total_pos, 1)
    df_gain["population"] = (df_gain.index + 1) / len(df_gain)

    gain_fig = go.Figure()
    gain_fig.add_trace(go.Scatter(x=df_gain["population"], y=df_gain["cum_gain"], mode="lines", name="Model"))
    gain_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    gain_fig.update_layout(title="Cumulative Gain", xaxis_title="Population", yaxis_title="Gain")
    sections.append(("Cumulative Gain Plot", _wrap_with_purpose("Cumulative Gain Plot", plotly_to_div(gain_fig), PURPOSE)))

    # 5) Dynamic threshold (interactive)
    auroc = float(roc_auc_score(np.asarray(y_test).reshape(-1), p))
    ll = float(log_loss(np.asarray(y_test).reshape(-1), np.asarray(y_proba)))
    br = float(brier_score_loss(np.asarray(y_test).reshape(-1), p))
    th_block = build_dynamic_threshold_block(y_true=np.asarray(y_test).reshape(-1), p=p, auroc=auroc, logloss=ll, brier=br)
    sections.append(("Dynamic Threshold Analysis (Interactive)", _wrap_with_purpose("Dynamic Threshold Analysis (Interactive)", th_block, PURPOSE)))

    # 6) Metrics
    core = {
        "Accuracy@0.5": float(accuracy_score(y_test, y_pred)),
        "Precision@0.5": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall@0.5": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1@0.5": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC@0.5": float(matthews_corrcoef(y_test, y_pred)),
        "AU-ROC": float(auroc),
        "Log Loss": float(ll),
        "Brier Score": float(br),
    }
    core_df = pd.DataFrame([{"metric": k, "value": v} for k, v in core.items()])
    sections.append(("Core Classification Metrics", _wrap_with_purpose("Core Classification Metrics", df_to_html_table(core_df), PURPOSE)))

    # 7) Classification report
    cr_df = _classification_report_table(y_test, y_pred)
    sections.append(("Classification Report", _wrap_with_purpose("Classification Report", df_to_html_table(cr_df), PURPOSE)))

    # 8) Calibration
    prob_true, prob_pred = calibration_curve(np.asarray(y_test).reshape(-1), p, n_bins=10, strategy="quantile")
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
    cal_fig.update_layout(title="Calibration Chart", xaxis_title="Pred prob", yaxis_title="Observed freq")
    sections.append(("Calibration", _wrap_with_purpose("Calibration", plotly_to_div(cal_fig), PURPOSE)))

    # 9) VIF + Corr + Phik
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xnum = X_test_df.select_dtypes(include=[np.number]).dropna()
        if Xnum.shape[1] >= 2:
            vals = Xnum.to_numpy()
            vifs = [{"feature": c, "VIF": float(variance_inflation_factor(vals, i))} for i, c in enumerate(Xnum.columns)]
            vif_df = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
            sections.append(("VIF (Multicollinearity)", _wrap_with_purpose("VIF (Multicollinearity)", df_to_html_table(vif_df), PURPOSE)))
        else:
            sections.append(("VIF (Multicollinearity)", _wrap_with_purpose("VIF (Multicollinearity)", "<pre class='code'>Not enough numeric features for VIF.</pre>", PURPOSE)))
    except Exception as e:
        sections.append(("VIF (Multicollinearity)", _wrap_with_purpose("VIF (Multicollinearity)", f"<pre class='code'>VIF skipped: {e}</pre>", PURPOSE)))

    corr = X_test_df.select_dtypes(include=[np.number]).corr()
    if corr.shape[0] > 1:
        corr_fig = px.imshow(corr, title="Correlation Heatmap (Numeric Features)")
        sections.append(("Correlation Heatmap", _wrap_with_purpose("Correlation Heatmap", plotly_to_div(corr_fig), PURPOSE)))
    else:
        sections.append(("Correlation Heatmap", _wrap_with_purpose("Correlation Heatmap", "<pre class='code'>Not enough numeric features.</pre>", PURPOSE)))

    if maybe_import_phik():
        try:
            import phik  # noqa
            tmp = X_test_df.copy()
            tmp["target"] = y_test
            phik_mat = tmp.phik_matrix()
            phik_fig = px.imshow(phik_mat, title="Phik Correlation Matrix")
            sections.append(("Phik Correlation Analysis", _wrap_with_purpose("Phik Correlation Analysis", plotly_to_div(phik_fig), PURPOSE)))
        except Exception as e:
            sections.append(("Phik Correlation Analysis", _wrap_with_purpose("Phik Correlation Analysis", f"<pre class='code'>Phik failed: {e}</pre>", PURPOSE)))
    else:
        sections.append(("Phik Correlation Analysis", _wrap_with_purpose("Phik Correlation Analysis", "<pre class='code'>phik not installed; skipped. (pip install phik)</pre>", PURPOSE)))

    html = wrap_html("01 – Model Performance (Classification)", sections)
    return save_html(out_dir / "01_model_performance.html", html)
