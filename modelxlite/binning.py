import numpy as np
import pandas as pd
from .utils import to_df

def bin_table_binary(y_true, y_prob, X_raw=None, bins=10) -> pd.DataFrame:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    if X_raw is not None:
        X_raw = to_df(X_raw)
        if len(X_raw) != len(y_true):
            raise ValueError("X_raw rows must match y_true length.")

    df = pd.DataFrame({"pred_prob": y_prob, "y": y_true})
    if X_raw is not None:
        df = pd.concat([df, X_raw.reset_index(drop=True)], axis=1)

    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    splits = np.array_split(df, bins)

    total_resp = df["y"].sum()
    rand_mean = df["y"].mean()

    raw_cols = list(X_raw.columns) if X_raw is not None else []
    rows = []
    cum_resp = 0
    cum_n = 0

    for i, chunk in enumerate(splits, start=1):
        n = len(chunk)
        if n == 0:
            continue

        responders = chunk["y"].sum()
        cum_resp += responders
        cum_n += n

        pred_mean = chunk["pred_prob"].mean()
        actual_mean = chunk["y"].mean()

        cum_precision = cum_resp / max(cum_n, 1)
        cum_recall = cum_resp / max(total_resp, 1)
        lift = cum_precision / max(rand_mean, 1e-12)

        row = {
            "BIN": i,
            "N": int(n),
            "PRED_MEAN": float(pred_mean),
            "ACTUAL_MEAN": float(actual_mean),
            "RESPONDERS": int(responders),
            "CUM_RESPONDERS": int(cum_resp),
            "CUM_PRECISION": float(cum_precision),
            "CUM_RECALL": float(cum_recall),
            "LIFT": float(lift),
            "PRED_MIN": float(chunk["pred_prob"].min()),
            "PRED_MAX": float(chunk["pred_prob"].max()),
        }

        for c in raw_cols:
            try:
                row[f"{c} MEAN"] = float(chunk[c].mean())
            except Exception:
                row[f"{c} MEAN"] = None

        rows.append(row)

    return pd.DataFrame(rows)

def ks_table_binary(y_true, y_prob, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=n_bins, labels=False) + 1

    out = []
    total_pos = (df["y"] == 1).sum()
    total_neg = (df["y"] == 0).sum()
    cum_pos = 0
    cum_neg = 0

    for d in range(1, n_bins + 1):
        sub = df[df["decile"] == d]
        pos = (sub["y"] == 1).sum()
        neg = (sub["y"] == 0).sum()
        cum_pos += pos
        cum_neg += neg

        tpr = cum_pos / max(total_pos, 1)
        fpr = cum_neg / max(total_neg, 1)
        ks = abs(tpr - fpr)

        out.append({
            "DECILE": int(d),
            "N": int(len(sub)),
            "POS": int(pos),
            "NEG": int(neg),
            "CUM_POS": int(cum_pos),
            "CUM_NEG": int(cum_neg),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "KS": float(ks),
            "P_MIN": float(sub["p"].min()),
            "P_MAX": float(sub["p"].max()),
            "P_MEAN": float(sub["p"].mean()),
        })
    return pd.DataFrame(out)
