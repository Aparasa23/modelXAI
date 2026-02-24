# html_utils.py
from __future__ import annotations

from pathlib import Path
from string import Template
from typing import List, Tuple, Union, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go


def df_to_html_table(df: pd.DataFrame, max_rows: int = 500, float_fmt: str = "{:.6f}") -> str:
    dfx = df.copy()
    if len(dfx) > max_rows:
        dfx = dfx.head(max_rows)

    for c in dfx.columns:
        if pd.api.types.is_float_dtype(dfx[c]) or pd.api.types.is_numeric_dtype(dfx[c]):
            def _fmt(x):
                if pd.isna(x):
                    return ""
                try:
                    return float_fmt.format(float(x))
                except Exception:
                    return str(x)
            dfx[c] = dfx[c].map(_fmt)

    return dfx.to_html(index=False, escape=False)


def plotly_to_div(fig: go.Figure) -> str:
    # Use CDN plotlyjs for lighter html
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _badge(status: str) -> str:
    s = (status or "INFO").strip().lower()
    if s not in {"ran", "skipped", "failed", "warn", "info"}:
        s = "info"
    return f"<span class='badge {s}'>{s.upper()}</span>"


def wrap_html(
    title: str,
    sections: List[Tuple[str, str]],
    statuses: Optional[Sequence[str]] = None,
    summary_items: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """
    sections: list of (heading, html_body)
    statuses: optional list parallel to sections with values in {RAN, SKIPPED, FAILED, WARN, INFO}
    summary_items: optional list of (key, value) displayed at top
    """
    statuses = list(statuses) if statuses is not None else ["INFO"] * len(sections)
    if len(statuses) < len(sections):
        statuses = statuses + ["INFO"] * (len(sections) - len(statuses))
    statuses = statuses[: len(sections)]

    # Build TOC + sections
    toc_items = []
    section_html = []
    for i, ((h, body), st) in enumerate(zip(sections, statuses), start=1):
        anchor = f"{h.lower().strip().replace(' ', '-').replace('/', '-')}-{i}"
        toc_items.append(f"<li><a href='#{anchor}'>{h}</a> {_badge(st)}</li>")
        section_html.append(
            f"""
<section class="report-section" id="{anchor}">
  <div class="section-header">
    <h2>{h}</h2>
    <div class="section-badges">{_badge(st)}</div>
  </div>
  <div class="section-body">
    {body}
  </div>
</section>
""".strip()
        )

    summary_html = ""
    if summary_items:
        rows = "".join([f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in summary_items])
        summary_html = f"""
<div class="summary">
  <h2>Run Summary</h2>
  <table class="summary-table">
    {rows}
  </table>
</div>
""".strip()

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #121212;
      --muted: #666;
      --border: #e6e6e6;
      --card: #fafafa;
      --shadow: 0 2px 10px rgba(0,0,0,0.06);

      --ran: #16a34a;
      --skipped: #64748b;
      --failed: #dc2626;
      --warn: #f59e0b;
      --info: #2563eb;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--fg);
    }}

    header {{
      padding: 18px 22px 0 22px;
      max-width: 1400px;
      margin: 0 auto;
    }}

    h1 {{
      margin: 0 0 8px 0;
      font-size: 22px;
      font-weight: 700;
    }}

    .subtitle {{
      margin: 0 0 10px 0;
      color: var(--muted);
      font-size: 13px;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 18px;
      padding: 18px 22px;
      max-width: 1400px;
      margin: 0 auto;
    }}

    .sidebar {{
      position: sticky;
      top: 12px;
      align-self: start;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 12px;
      box-shadow: var(--shadow);
      height: fit-content;
    }}

    .sidebar-title {{
      font-weight: 700;
      margin-bottom: 10px;
      font-size: 14px;
    }}

    .toc {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 13px;
    }}

    .toc a {{
      text-decoration: none;
      color: var(--fg);
    }}

    .toc a:hover {{
      text-decoration: underline;
    }}

    .main {{
      min-width: 0;
    }}

    .summary {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 14px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }}

    .summary h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
    }}

    .summary-table td {{
      border: 1px solid var(--border);
      padding: 8px;
      font-size: 13px;
    }}

    .report-section {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 14px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }}

    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 12px;
    }}

    .section-header h2 {{
      margin: 0;
      font-size: 16px;
    }}

    .section-body {{
      font-size: 13px;
    }}

    .code {{
      background: #f6f6f6;
      padding: 12px;
      border-radius: 10px;
      overflow-x: auto;
      border: 1px solid var(--border);
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
    }}

    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      font-size: 12px;
      vertical-align: top;
    }}
    th {{
      background: #f2f2f2;
      position: sticky;
      top: 0;
      z-index: 1;
    }}

    .badge {{
      display: inline-block;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      color: white;
      margin-left: 6px;
      line-height: 18px;
      white-space: nowrap;
    }}
    .badge.ran {{ background: var(--ran); }}
    .badge.skipped {{ background: var(--skipped); }}
    .badge.failed {{ background: var(--failed); }}
    .badge.warn {{ background: var(--warn); color: #111; }}
    .badge.info {{ background: var(--info); }}

    details.collapsible {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 10px;
      background: var(--card);
      margin-top: 10px;
    }}
    details.collapsible summary {{
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
    }}
    .collapsible-body {{
      margin-top: 10px;
    }}

    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: relative;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="subtitle">Generated by ModelXAI Lite • Static HTML report</p>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-title">Contents</div>
      <ul class="toc">
        {''.join(toc_items)}
      </ul>
    </aside>

    <main class="main">
      {summary_html}
      {''.join(section_html)}
    </main>
  </div>
</body>
</html>
""".strip()



from pathlib import Path
from typing import Union

from pathlib import Path
from typing import Union

def save_html(out_path: Union[str, Path], html: str) -> Path:
    """
    Save HTML and return a Path object (not str).
    This preserves runner.py expectations (read_text, etc.).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def safe_write_text(path: Union[str, Path], text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
