import json
from pathlib import Path
import pandas as pd
from .utils import to_df
from .counterfactuals_engine import CounterfactualConfig, precompute_cases

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
    rules.append(("overview", "Purpose: Explain what counterfactuals are and how to use them safely. How it helps: converts 'why' into 'what can change' (actionable recourse). What to watch for: unrealistic suggestions."))
    rules.append(("case selector", "Purpose: Choose a specific case to generate actionable alternatives. How it helps: interactive exploration for business and risk stakeholders."))
    rules.append(("ranked", "Purpose: List the best alternative scenarios that flip the decision with minimal change. How it helps: shows cheapest paths to approval/desired outcome."))
    rules.append(("row inspector", "Purpose: Side-by-side comparison of original vs counterfactual values. How it helps: highlights exactly which fields change and by how much."))
    rules.append(("constraints", "Purpose: Explain which fields are immutable and which directions are allowed. How it helps: ensures recommendations are feasible and compliant."))
    rules.append(("distance", "Purpose: Define what “minimal change” means (L1/L2/normalized). How it helps: aligns suggestions to business cost."))
    rules.append(("warnings", "Purpose: Show caveats: correlation vs causation, policy constraints, data quality. How it helps: prevents over-trust and misuse."))

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

def _safe_json(obj) -> str:
    """Safe JSON for inline <script> embedding (prevents </script> termination)."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</script>", "<\\/script>").replace("</SCRIPT>", "<\\/SCRIPT>")


def _build_interactive_block(payload: dict) -> str:
    """Readability-first interactive counterfactual UI (offline HTML)."""
    js_payload = _safe_json(payload)

    return f"""
<div class="subheader">
  <div><b>How to study this section</b></div>
  <ul class="guide">
    <li><b>Step 1:</b> Pick a case. Start with <b>Actionable + Feasible</b> rows.</li>
    <li><b>Step 2:</b> Prefer the <b>lowest distance</b> and <b>fewest features_changed</b>.</li>
    <li><b>Step 3:</b> Use <b>Row inspector</b> to read the exact “Original → Counterfactual” changes.</li>
    <li><b>Tip:</b> If tiny changes flip the outcome, the case is near the decision boundary (borderline).</li>
  </ul>
</div>

<!-- Stacked layout for readability -->
<div class="cf-stack">

  <div class="cf-card">
    <div class="cf-card-title">1) Choose a case</div>
    <div class="row">
      <div class="col">
        <div class="muted tiny">Case selector</div>
        <select id="cfCaseSelect" class="cf-select"></select>
      </div>
      <div class="col meta">
        <div><span class="muted">Row index:</span> <span id="cfIdx"></span></div>
        <div><span class="muted">Prediction:</span> <span id="cfPred"></span></div>
        <div><span class="muted">Probabilities:</span> <span id="cfProba"></span></div>
      </div>
    </div>

    <div class="cf-card-title" style="margin-top:12px;">Original row</div>
    <div class="tablewrap" style="max-height:360px;">
      <div id="cfOriginal"></div>
    </div>
  </div>

  <div class="cf-card">
    <div class="cf-card-title">2) Pick a generation method + filters</div>

    <div class="controls">
      <div class="control grow">
        <div class="muted tiny">Method</div>
        <div class="cf-tabs">
          <button class="cf-tab" data-method="random" id="tab-random">Random</button>
          <button class="cf-tab" data-method="genetic" id="tab-genetic">Genetic</button>
        </div>
      </div>

      <div class="control">
        <label class="chk">
          <input type="checkbox" id="toggleDeltaOnly" checked />
          <span>Delta-only columns</span>
        </label>
        <label class="chk">
          <input type="checkbox" id="toggleOnlyGood" />
          <span>Only actionable + feasible</span>
        </label>
      </div>
    </div>

    <div class="controls">
      <div class="control grow">
        <div class="muted tiny">Feature filter (deltas)</div>
        <select id="cfFeatureSelect" class="cf-select" multiple size="6"></select>
        <div class="muted tiny">Tip: Ctrl/Cmd-click to select multiple. Leave empty to show all deltas.</div>
      </div>

      <div class="control grow">
        <div class="muted tiny">Method summary</div>
        <div id="cfMethodMeta" class="cf-mini" style="margin-top:0;"></div>
      </div>
    </div>
  </div>

  <div class="cf-card">
    <div class="cf-card-title">3) Ranked counterfactual table</div>
    <div class="muted tiny" style="margin-bottom:8px;">
      Scan here for the best option (actionable + feasible, lowest distance).
    </div>
    <div id="cfTable"></div>
  </div>

  <div class="cf-card">
    <div class="cf-card-title">4) Row inspector (easy reading)</div>
    <div class="muted tiny">Select one CF row and read “Original → Counterfactual” per feature.</div>
    <div class="row">
      <div class="col">
        <select id="cfRowSelect" class="cf-select"></select>
      </div>
    </div>
    <div id="cfCompare" class="tablewrap" style="max-height:520px; margin-top:10px;"></div>
  </div>

</div>

<script>
const CF_PAYLOAD = {js_payload};

function esc(x) {{
  if (x === null || x === undefined) return "";
  return String(x).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}}

function isMeaningfulDelta(v) {{
  if (v === null || v === undefined) return false;
  if (typeof v === "number") return Math.abs(v) > 1e-12;
  const s = String(v).trim().toLowerCase();
  if (s === "" || s === "0" || s === "0.0" || s === "nan" || s === "none") return false;
  return true;
}}

function originalTable(obj) {{
  const keys = Object.keys(obj);
  let html = "<table><thead><tr><th>feature</th><th>value</th></tr></thead><tbody>";
  for (const k of keys) {{
    html += `<tr><td>${{esc(k)}}</td><td>${{esc(obj[k])}}</td></tr>`;
  }}
  html += "</tbody></table>";
  return html;
}}

function getDeltaColumns(records, deltaOnly) {{
  if (!records || records.length === 0) return [];
  const cols = Object.keys(records[0]).filter(c => c.startsWith("Δ "));
  if (!deltaOnly) return cols;
  return cols.filter(c => records.some(r => isMeaningfulDelta(r[c])));
}}

function getSelectedFeatures() {{
  const sel = document.getElementById("cfFeatureSelect");
  const out = [];
  for (const opt of sel.selectedOptions) out.push(opt.value);
  return out;
}}

function tableFromRecords(records, opts) {{
  opts = opts || {{}};
  const deltaOnly = (opts.deltaOnly !== undefined) ? opts.deltaOnly : true;
  const onlyGood = !!opts.onlyGood;
  const selectedFeatures = opts.selectedFeatures || null;

  if (!records || records.length === 0) return "<pre class='code'>No counterfactuals generated for this method.</pre>";

  let rows = records.slice();
  if (onlyGood) {{
    rows = rows.filter(r => r.actionable !== false && r.feasible !== false);
  }}
  if (rows.length === 0) {{
    return "<pre class='code'>No rows match the filter (actionable + feasible).</pre>";
  }}

  const metaCols = ["features_changed","normalized_l1_distance","l1_distance","actionable","feasible"];
  const colsAll = Object.keys(rows[0] || {{}});
  let deltaCols = getDeltaColumns(rows, deltaOnly);

  if (selectedFeatures && selectedFeatures.length > 0) {{
    const selectedDeltaCols = selectedFeatures.map(f => "Δ " + f);
    deltaCols = deltaCols.filter(c => selectedDeltaCols.includes(c));
  }}

  const cols = metaCols.filter(c => colsAll.includes(c)).concat(deltaCols);

  let html = "<div class='tablewrap'><table><thead><tr>";
  for (const c of cols) html += `<th>${{esc(c)}}</th>`;
  html += "</tr></thead><tbody>";

  for (const r of rows) {{
    html += "<tr>";
    for (const c of cols) {{
      let v = r[c];
      let cls = "";
      if (c === "actionable" && r[c] === false) cls = "bad";
      if (c === "feasible" && r[c] === false) cls = "bad";
      if (c.startsWith("Δ ") && isMeaningfulDelta(v)) cls = (cls ? cls + " " : "") + "changed";

      if (typeof v === "number") {{
        v = (Math.abs(v) >= 1000 ? v.toFixed(2) : v.toFixed(4));
      }}
      html += `<td class="${{cls}}">${{esc(v)}}</td>`;
    }}
    html += "</tr>";
  }}

  html += "</tbody></table></div>";
  return html;
}}

function compareTable(original, cfRow) {{
  const keys = Object.keys(original);
  let html = "<table><thead><tr><th>feature</th><th>original</th><th>counterfactual</th><th>Δ</th></tr></thead><tbody>";
  for (const k of keys) {{
    if (!(k in cfRow)) continue;
    const a = original[k];
    const b = cfRow[k];
    const dkey = "Δ " + k;
    const d = (dkey in cfRow) ? cfRow[dkey] : "";
    const changed = isMeaningfulDelta(d);
    html += `<tr class="${{changed ? "rowchanged" : ""}}"><td>${{esc(k)}}</td><td>${{esc(a)}}</td><td>${{esc(b)}}</td><td>${{esc(d)}}</td></tr>`;
  }}
  html += "</tbody></table>";
  return html;
}}

let currentMethod = "random";

function renderCase(i) {{
  const cases = CF_PAYLOAD.cases || [];
  const c = cases[i];
  if (!c) return;

  document.getElementById("cfIdx").textContent = c.index;

  const predCtx = c.pred_ctx || {{}};
  document.getElementById("cfPred").textContent = (predCtx.pred ?? "");
  document.getElementById("cfProba").textContent = predCtx.proba ? JSON.stringify(predCtx.proba) : "—";

  document.getElementById("cfOriginal").innerHTML = originalTable(c.cf.query || {{}});

  renderMethod(i, currentMethod, true);
}}

function populateFeatureSelect(records) {{
  const sel = document.getElementById("cfFeatureSelect");
  const current = new Set(getSelectedFeatures());
  sel.innerHTML = "";

  if (!records || records.length === 0) return;

  const cols = Object.keys(records[0]).filter(c => c.startsWith("Δ "));
  const feats = cols.map(c => c.slice(2)).sort((a,b) => a.localeCompare(b));

  for (const f of feats) {{
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    if (current.has(f)) opt.selected = true;
    sel.appendChild(opt);
  }}
}}

function populateRowSelect(records) {{
  const sel = document.getElementById("cfRowSelect");
  sel.innerHTML = "";
  if (!records || records.length === 0) {{
    const opt = document.createElement("option");
    opt.value = "0";
    opt.textContent = "No counterfactual rows";
    sel.appendChild(opt);
    return;
  }}

  for (let i=0; i<records.length; i++) {{
    const r = records[i];
    const dist = (r.normalized_l1_distance !== undefined && r.normalized_l1_distance !== null) ? r.normalized_l1_distance : r.l1_distance;
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `CF #${{i}} (changed=${{r.features_changed}}, dist=${{dist}})`;
    sel.appendChild(opt);
  }}
}}

function renderMethod(i, method, refreshPickers) {{
  currentMethod = method;

  document.querySelectorAll(".cf-tab").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.querySelector(`.cf-tab[data-method="${{method}}"]`);
  if (activeBtn) activeBtn.classList.add("active");

  const cases = CF_PAYLOAD.cases || [];
  const c = cases[i];
  const m = (c && c.cf && c.cf.methods) ? c.cf.methods[method] : null;

  const meta = (m && m.meta) ? m.meta : {{}};
  let metaHtml = "<div><span class='muted'>Method:</span> " + esc(meta.method || method) + "</div>";
  if (meta.error) {{
    metaHtml += "<div class='bad'><b>Error:</b> " + esc(meta.error) + "</div>";
  }} else {{
    metaHtml += "<div><span class='muted'># CFs:</span> " + esc(meta.n) + "</div>";
    metaHtml += "<div><span class='muted'>Best distance:</span> " + esc(meta.best_distance) + "</div>";
    metaHtml += "<div><span class='muted'>Best actionable:</span> " + esc(meta.best_actionable) + "</div>";
    metaHtml += "<div><span class='muted'>Best feasible:</span> " + esc(meta.best_feasible) + "</div>";
  }}
  document.getElementById("cfMethodMeta").innerHTML = metaHtml;

  const toggleDeltaOnly = document.getElementById("toggleDeltaOnly").checked;
  const toggleOnlyGood = document.getElementById("toggleOnlyGood").checked;
  const selectedFeatures = getSelectedFeatures();

  const records = (m && m.table) ? m.table : [];

  if (refreshPickers) {{
    populateFeatureSelect(records);
    populateRowSelect(records);
  }}

  document.getElementById("cfTable").innerHTML = tableFromRecords(records, {{
    deltaOnly: toggleDeltaOnly,
    onlyGood: toggleOnlyGood,
    selectedFeatures: selectedFeatures
  }});

  const selRow = document.getElementById("cfRowSelect");
  const rowIdx = selRow && selRow.value ? parseInt(selRow.value, 10) : 0;
  const original = (c && c.cf) ? (c.cf.query || {{}}) : {{}};
  const cfRow = records[rowIdx] || (records[0] || {{}});
  document.getElementById("cfCompare").innerHTML = compareTable(original, cfRow);
}}

function init() {{
  const sel = document.getElementById("cfCaseSelect");
  const cases = CF_PAYLOAD.cases || [];

  sel.innerHTML = "";
  for (let i=0; i<cases.length; i++) {{
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Case #${{i}} (row index: ${{cases[i].index}})`;
    sel.appendChild(opt);
  }}

  sel.addEventListener("change", () => renderCase(parseInt(sel.value, 10)));

  document.querySelectorAll(".cf-tab").forEach(btn => {{
    btn.addEventListener("click", () => {{
      renderMethod(parseInt(sel.value, 10), btn.dataset.method, true);
    }});
  }});

  document.getElementById("toggleDeltaOnly").addEventListener("change", () => {{
    renderMethod(parseInt(sel.value, 10), currentMethod, false);
  }});
  document.getElementById("toggleOnlyGood").addEventListener("change", () => {{
    renderMethod(parseInt(sel.value, 10), currentMethod, false);
  }});
  document.getElementById("cfFeatureSelect").addEventListener("change", () => {{
    renderMethod(parseInt(sel.value, 10), currentMethod, false);
  }});
  document.getElementById("cfRowSelect").addEventListener("change", () => {{
    renderMethod(parseInt(sel.value, 10), currentMethod, false);
  }});

  if (cases.length > 0 && cases[0].cf && cases[0].cf.methods) {{
    const keys = Object.keys(cases[0].cf.methods);
    if (keys.includes("random")) currentMethod = "random";
    else if (keys.includes("genetic")) currentMethod = "genetic";
    else currentMethod = keys[0];
  }}

  if (cases.length > 0) renderCase(0);
}}
init();
</script>

<style>
/* Neutral, non-blue styling; stacked layout for readability */
.subheader {{
  display:flex;
  flex-direction:column;
  gap:8px;
  padding: 12px 14px;
  background: #f7f7f7;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
  margin-bottom: 14px;
}}
.guide {{ margin: 0; padding-left: 18px; }}
.guide li {{ margin: 4px 0; }}

.muted {{ color:#666; }}
.tiny {{ font-size: 12px; }}
.bad {{ color:#b00020; }}

.cf-stack {{
  display:flex;
  flex-direction:column;
  gap: 14px;
}}

.cf-card {{
  background:#fff;
  border: 1px solid #e6e6e6;
  border-radius: 14px;
  padding: 14px;
}}

.cf-card-title {{
  font-weight: 800;
  margin-bottom: 8px;
}}

.row {{
  display:flex;
  gap: 12px;
  align-items: flex-start;
  flex-wrap: wrap;
}}

.col {{
  flex: 1 1 320px;
  min-width: 280px;
}}
.col.meta {{
  flex: 1 1 260px;
  min-width: 240px;
  background: #fafafa;
  border: 1px solid #e8e8e8;
  border-radius: 12px;
  padding: 10px 12px;
}}

.controls {{
  display:flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  align-items: flex-end;
  margin: 10px 0 8px 0;
}}

.control {{ min-width: 260px; }}
.control.grow {{ flex: 1 1 360px; }}

.chk {{
  display:flex;
  gap: 8px;
  align-items:center;
  user-select:none;
  margin-right: 10px;
}}

.cf-select {{
  width: 100%;
  padding: 10px 10px;
  border-radius: 10px;
  border: 1px solid #dcdcdc;
  background: #fff;
  outline: none;
}}

.cf-mini {{
  margin-top: 6px;
  display:flex;
  flex-direction:column;
  gap:4px;
  font-size: 14px;
}}

.cf-tabs {{
  display:flex;
  gap: 8px;
  margin-top: 6px;
}}

.cf-tab {{
  border: 1px solid #dcdcdc;
  background: #fafafa;
  padding: 8px 10px;
  border-radius: 10px;
  cursor: pointer;
}}

.cf-tab.active {{
  background: #111;
  color:#fff;
  border-color:#111;
}}

.tablewrap {{
  overflow:auto;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
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
td.changed {{ font-weight: 800; }}
tr.rowchanged td {{ background: #fffdf0; }}

pre.code {{
  background:#f3f3f3;
  padding: 10px;
  border-radius: 10px;
  overflow-x:auto;
}}
</style>
"""


def build_report3_counterfactuals(out_dir, cfg, model, train_df, target_name, X_test, query_index=0):
    """Enterprise-grade counterfactual report with readability-first UI."""
    sections = []

    if train_df is None:
        sections.append(("Error", "<pre class='code'>train_df is required for counterfactuals.</pre>"))

        html = wrap_html("03 – Counterfactual Explanations", sections)
        return _save_html_compat(out_dir / "03_counterfactuals.html", html)

    X = to_df(X_test)

    # Optional config (non-breaking)
    immutable = tuple(getattr(cfg, "immutable_features", ())) or ()
    permitted_range = getattr(cfg, "permitted_range", None)
    direction_constraints = getattr(cfg, "direction_constraints", None)

    # Precompute indices for selector; ensure query_index is included
    max_cases = int(getattr(cfg, "counterfactual_max_cases", 25) or 25)
    max_cases = max(1, min(max_cases, len(X)))
    indices = list(range(max_cases))
    qi = int(max(0, min(int(query_index), len(X) - 1)))
    if qi not in indices:
        indices[0] = qi

    cf_cfg = CounterfactualConfig(
        total_cfs=int(getattr(cfg, "counterfactual_total_cfs", 3) or 3),
        methods=tuple(getattr(cfg, "counterfactual_methods", ("random", "genetic"))),
        max_cases=max_cases,
        seed=int(getattr(cfg, "seed", 7) or 7),
        immutable_features=immutable,
        permitted_range=permitted_range,
        direction_constraints=direction_constraints,
        distance=str(getattr(cfg, "counterfactual_distance", "normalized_l1") or "normalized_l1"),
    )

    payload = precompute_cases(
        cfg=cf_cfg,
        model=model,
        train_df=train_df,
        target_name=target_name,
        X=X,
        problem_type=str(cfg.problem_type),
        query_indices=indices,
    )

    sections.append(("Counterfactual Recourse – Interactive Case Selector", _build_interactive_block(payload)))

    sections.append((
        "Interpretation notes (practical)",
        """
        <ul>
          <li><b>Actionable</b>: immutable features did not change (e.g., age, gender).</li>
          <li><b>Feasible</b>: changes respect your permitted ranges / direction constraints (if provided).</li>
          <li><b>Distance</b>: approximates “how big the change is” (lower is better).</li>
          <li><b>features_changed</b>: prefer fewer changes for a simpler recourse story.</li>
        </ul>
        """,
    ))
    sections = _apply_section_explanations(sections)

    sections = _apply_section_explanations(sections)

    html = wrap_html("03 – Counterfactual Explanations", sections)
    return _save_html_compat(out_dir / "03_counterfactuals.html", html)