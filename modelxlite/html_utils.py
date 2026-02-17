from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import plotly.graph_objects as go

def df_to_html_table(df: pd.DataFrame, max_rows=500, float_fmt="{:.6f}") -> str:
    dfx = df.copy()
    if len(dfx) > max_rows:
        dfx = dfx.head(max_rows)

    for c in dfx.columns:
        if pd.api.types.is_float_dtype(dfx[c]):
            dfx[c] = dfx[c].map(lambda x: "" if pd.isna(x) else float_fmt.format(x))

    return dfx.to_html(index=False, escape=False)

def plotly_to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def wrap_html(title: str, sections: List[Tuple[str, str]]) -> str:
    sec_html = ""
    for h, body in sections:
        sec_html += f"<h2>{h}</h2>\n<div class='section'>{body}</div>\n"

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 22px; }}
    .section {{ margin: 14px 0; }}
    .code {{ background: #f6f6f6; padding: 10px; border-radius: 8px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 12px; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {sec_html}
</body>
</html>
""".strip()

def save_html(path: Union[str, Path], html: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
