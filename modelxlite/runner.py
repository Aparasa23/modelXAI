from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd

from .config import modelxReportConfig
from .utils import to_df
from .html_utils import wrap_html, save_html, df_to_html_table
from .binning import bin_table_binary
from .report1_model_performance import build_report1_classification
from .report2_interpretability import build_report2_interpretability
from .report3_counterfactuals import build_report3_counterfactuals
from .report4_drift_quality import build_report4_drift_quality

class modelxLiteProject:
    def __init__(self, cfg: modelxReportConfig):
        self.cfg = cfg
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.target_name: str = "target"

    def fit(self, model, X_train, y_train, train_df: Optional[pd.DataFrame]=None, target_name="target"):
        self.model = model
        self.target_name = target_name

        Xtr = to_df(X_train)
        self.feature_names = list(Xtr.columns)

        if train_df is not None:
            self.train_df = train_df.copy()
        else:
            self.train_df = Xtr.copy()
            self.train_df[target_name] = np.asarray(y_train).reshape(-1)

        return self

    def generate(
        self,
        output_dir: Union[str, Path],
        run_name: str,
        X_test,
        y_test,
        X_train=None,
        y_train=None,
        X_train_raw=None,
        X_test_raw=None,
        which_reports: Union[str, List[int]] = "all",  # "all" or [1,2,3,4]
        report_on: str = "test",                       # for distribution tables inside report1
        shap_local_index: int = 0,
        pdp_features: Optional[List[Union[int, str]]] = None,
        drift_ref=None,
        drift_cur=None,
    ) -> Dict[str, Path]:

        out_dir = Path(output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if which_reports == "all":
            selected = [1, 2, 3, 4]
        else:
            selected = sorted(set(which_reports))

        Xte = to_df(X_test, feature_names=self.feature_names)
        yte = np.asarray(y_test).reshape(-1)

        saved: Dict[str, Path] = {}

        # Predictions
        if self.cfg.problem_type == "classification":
            y_pred = self.model.predict(Xte)
            y_proba = self.model.predict_proba(Xte) if hasattr(self.model, "predict_proba") else None
        else:
            y_pred = np.asarray(self.model.predict(Xte)).reshape(-1)
            y_proba = None

        # 1) Model performance
        if 1 in selected:
            if self.cfg.problem_type == "classification":
                # Build report #1 with requested ordering
                path1 = build_report1_classification(
                    out_dir=out_dir,
                    X_test_df=Xte,
                    y_test=yte,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    bins=self.cfg.bins,
                    report_on=report_on,
                    X_train=X_train,
                    y_train=y_train,
                    X_train_raw=X_train_raw,
                    X_test_raw=X_test_raw
                )

                # If user requested TRAIN distribution inside report #1, append it in correct place:
                # They want: Distribution (TEST/TRAIN if chosen) as the FIRST section.
                # Since build_report1_classification currently adds TEST distribution only,
                # we rebuild the first section here if train is needed.
                if report_on in ("train", "both") and X_train is not None and y_train is not None and y_proba is not None:
                    Xtr = to_df(X_train, feature_names=self.feature_names)
                    ytr = np.asarray(y_train).reshape(-1)
                    p_tr = self.model.predict_proba(Xtr)[:, 1]

                    bin_label = "Decile" if self.cfg.bins == 10 else "Percentile" if self.cfg.bins == 100 else f"Bin({self.cfg.bins})"
                    train_bins = bin_table_binary(ytr, p_tr, X_raw=X_train_raw, bins=self.cfg.bins).rename(columns={"BIN": bin_label.upper()})

                    # Inject TRAIN table after TEST table by rewriting HTML file
                    html = path1.read_text(encoding="utf-8")
                    marker = f"<h2>{bin_label} Distribution Report (TEST)</h2>"
                    insert = f"<h2>{bin_label} Distribution Report (TRAIN)</h2>\n<div class='section'>{df_to_html_table(train_bins)}</div>\n"

                    # insert after the TEST section block
                    # safest: insert after the closing div of TEST block (first occurrence)
                    idx = html.find(marker)
                    if idx != -1:
                        # find end of the first section div after marker
                        end_div = html.find("</div>", idx)
                        if end_div != -1:
                            end_div = html.find("</div>", end_div + 1)  # move to the outer section end
                        if end_div != -1:
                            html = html[:end_div+6] + "\n" + insert + html[end_div+6:]
                            path1.write_text(html, encoding="utf-8")

                saved["01_model_performance"] = path1
            else:
                # Regression report #1 not included in this snippet; if you want same style ordering for regression,
                # we can add a regression-specific report1 module similarly.
                # For now, write a simple placeholder.
                html = wrap_html("01 – Model Performance (Regression)", [
                    ("Note", "<pre class='code'>Regression ordering module can be added similarly.</pre>")
                ])
                saved["01_model_performance"] = save_html(out_dir / "01_model_performance.html", html)

        # 2) Interpretability
        if 2 in selected:
            path2 = build_report2_interpretability(
                out_dir=out_dir,
                cfg=self.cfg,
                model=self.model,
                X_test=Xte,
                feature_names=self.feature_names,
                shap_local_index=shap_local_index,
                pdp_features=pdp_features
            )
            saved["02_interpretability"] = path2

        # 3) Counterfactuals
        if 3 in selected:
            path3 = build_report3_counterfactuals(
                out_dir=out_dir,
                cfg=self.cfg,
                model=self.model,
                train_df=self.train_df,
                target_name=self.target_name,
                X_test=Xte,
                query_index=shap_local_index
            )
            saved["03_counterfactuals"] = path3

        # 4) Drift & Quality
        if 4 in selected:
            path4 = build_report4_drift_quality(
                out_dir=out_dir,
                X_eval=Xte,
                feature_names=self.feature_names,
                drift_ref=drift_ref,
                drift_cur=drift_cur,
                bins=self.cfg.bins
            )
            saved["04_drift_quality"] = path4

        return saved
