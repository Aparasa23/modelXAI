import numpy as np
import pandas as pd

def to_df(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)

def safe_mape(y_true, y_pred, eps=1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true, y_pred, eps=1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def psi(expected: pd.Series, actual: pd.Series, buckets=10, eps=1e-6) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    quantiles = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return float("nan")

    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

def maybe_import_phik() -> bool:
    try:
        import phik  # noqa
        return True
    except Exception:
        return False

def maybe_import_lime():
    try:
        import lime
        return True
    except:
        return False


def maybe_import_eli5():
    try:
        import eli5
        return True
    except:
        return False
