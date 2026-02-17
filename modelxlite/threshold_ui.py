import numpy as np
import pandas as pd
import json
import math

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef,
    hamming_loss, jaccard_score, zero_one_loss
)
import plotly.graph_objects as go


def _class_report_rows(y_true, y_hat):
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    labels = [0, 1]

    rows = []
    for cls in labels:
        tp = int(((y_true == cls) & (y_hat == cls)).sum())
        fp = int(((y_true != cls) & (y_hat == cls)).sum())
        fn = int(((y_true == cls) & (y_hat != cls)).sum())
        support = int((y_true == cls).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

        rows.append({
            "class": int(cls),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        })

    acc = float(accuracy_score(y_true, y_hat))
    rows.append({
        "class": "accuracy",
        "precision": None,   # ✅ JSON-safe
        "recall": None,      # ✅ JSON-safe
        "f1": acc,
        "support": int(len(y_true)),
    })
    return rows


def build_dynamic_threshold_block(
    y_true,
    p,
    auroc=None,
    logloss=None,
    brier=None,
    thresholds=None
) -> str:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    if thresholds is None:
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 3)

    core_rows = []
    classrep_rows = []

    for t in thresholds:
        y_hat = (p >= t).astype(int)

        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1v = f1_score(y_true, y_hat, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_hat)
        ham = hamming_loss(y_true, y_hat)
        jac = jaccard_score(y_true, y_hat, average="binary", zero_division=0)
        jac_loss = 1.0 - jac
        zol = zero_one_loss(y_true, y_hat)

        core_rows.append({
            "threshold": float(t),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1v),
            "MCC": float(mcc),
            "Hamming Loss": float(ham),
            "Jaccard Loss": float(jac_loss),
            "Zero-One Loss": float(zol),
        })

        classrep_rows.append(_class_report_rows(y_true, y_hat))

    core_df = pd.DataFrame(core_rows)

    # Curves plot (top)
    fig = go.Figure()
    for col in ["Accuracy", "Precision", "Recall", "F1", "MCC"]:
        fig.add_trace(go.Scatter(x=core_df["threshold"], y=core_df[col], mode="lines", name=col))
    fig.update_layout(
        title="Dynamic Threshold Analysis",
        xaxis_title="Threshold",
        yaxis_title="Metric value",
        height=420
    )
    curves_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Probability metrics (static)
    prob_metrics_html = "<div class='note'><b>Probability-based metrics (threshold-independent):</b><br/>"
    if auroc is not None:
        prob_metrics_html += f"AUROC: <b>{float(auroc):.6f}</b><br/>"
    if logloss is not None:
        prob_metrics_html += f"Log Loss: <b>{float(logloss):.6f}</b><br/>"
    if brier is not None:
        prob_metrics_html += f"Brier Score: <b>{float(brier):.6f}</b><br/>"
    prob_metrics_html += "</div>"

    # ✅ SAFE JSON serialization
    def _clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        return obj

    core_json = json.dumps(core_rows, default=_clean_for_json)
    classrep_json = json.dumps(classrep_rows, default=_clean_for_json)

    core_table_html = """
    <table id="coreMetricsTable">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody></tbody>
    </table>
    """

    cls_table_html = """
    <table id="classReportTable">
      <thead>
        <tr><th>class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr>
      </thead>
      <tbody></tbody>
    </table>
    """

    block = f"""
    <div class="section">

      {curves_div}

      <div style="margin-top:12px;">
        <label><b>Threshold</b></label>
        <input id="thrSlider" type="range" min="0" max="{len(core_rows)-1}" value="0" step="1" style="width:100%;">
        <div style="margin-top:6px;">
          Selected threshold: <b><span id="thrValue"></span></b>
        </div>
      </div>

      <div style="margin-top:14px;">
        {prob_metrics_html}
      </div>

      <h3 style="margin-top:16px;">Core Classification Metrics</h3>
      {core_table_html}

      <h3 style="margin-top:16px;">Classification Report</h3>
      {cls_table_html}

    </div>

    <script>
      const CORE = {core_json};
      const CLASSREP = {classrep_json};

      function fmt(x) {{
        if (x === null || x === undefined) return "";
        if (typeof x === "number") return x.toFixed(6);
        return String(x);
      }}

      function setCoreTable(i) {{
        const r = CORE[i];
        const tbody = document.querySelector("#coreMetricsTable tbody");
        tbody.innerHTML = "";

        const ordered = [
          ["Accuracy", r["Accuracy"]],
          ["Precision", r["Precision"]],
          ["Recall", r["Recall"]],
          ["F1", r["F1"]],
          ["MCC", r["MCC"]],
          ["Hamming Loss", r["Hamming Loss"]],
          ["Jaccard Loss", r["Jaccard Loss"]],
          ["Zero-One Loss", r["Zero-One Loss"]],
        ];

        for (const [k, v] of ordered) {{
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${{k}}</td><td>${{fmt(v)}}</td>`;
          tbody.appendChild(tr);
        }}
      }}

      function setClassReport(i) {{
        const rows = CLASSREP[i];
        const tbody = document.querySelector("#classReportTable tbody");
        tbody.innerHTML = "";

        for (const row of rows) {{
          const tr = document.createElement("tr");
          tr.innerHTML =
            `<td>${{row["class"]}}</td>` +
            `<td>${{fmt(row["precision"])}}</td>` +
            `<td>${{fmt(row["recall"])}}</td>` +
            `<td>${{fmt(row["f1"])}}</td>` +
            `<td>${{row["support"]}}</td>`;
          tbody.appendChild(tr);
        }}
      }}

      function updateAll(i) {{
        const t = CORE[i]["threshold"];
        document.getElementById("thrValue").textContent = t.toFixed(3);
        setCoreTable(i);
        setClassReport(i);
      }}

      const slider = document.getElementById("thrSlider");
      slider.addEventListener("input", (e) => {{
        updateAll(parseInt(e.target.value));
      }});

      // init near 0.50
      let initIndex = 0;
      let bestDiff = 1e9;
      for (let k = 0; k < CORE.length; k++) {{
        const d = Math.abs(CORE[k]["threshold"] - 0.5);
        if (d < bestDiff) {{
          bestDiff = d;
          initIndex = k;
        }}
      }}
      slider.value = initIndex;
      updateAll(initIndex);
    </script>
    """
    return block
