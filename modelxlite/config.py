from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class modelxReportConfig:
    # Core
    problem_type: str                 # "classification" or "regression"
    bins: int = 10                   # 10=deciles, 100=percentiles
    shap_sample: int = 2000

    # -------------------------
    # Counterfactuals (Report 3)
    # -------------------------

    # How many test rows to precompute for interactive selector
    counterfactual_max_cases: int = 25

    # How many CFs per case
    counterfactual_total_cfs: int = 3

    # DiCE methods
    counterfactual_methods: Tuple[str, ...] = ("random", "genetic")

    # Distance metric for ranking
    counterfactual_distance: str = "normalized_l1"   # or "l1"

    # Actionability / feasibility
    immutable_features: Tuple[str, ...] = ()

    # Numeric bounds per feature: {"income": (30000, None)}
    permitted_range: Optional[Dict[str, tuple]] = None

    # Monotonic constraints: {"age": "increase_only"}
    direction_constraints: Optional[Dict[str, str]] = None

    # Random seed for reproducibility
    seed: int = 7