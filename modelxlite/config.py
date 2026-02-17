from dataclasses import dataclass

@dataclass
class modelxReportConfig:
    problem_type: str  # "classification" or "regression"
    bins: int = 10     # 10=deciles, 100=percentiles
    shap_sample: int = 2000
