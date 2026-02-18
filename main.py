import os
import json
import tempfile
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import joblib

from google.cloud import storage

# Your package (must be included in deployment OR installed from a wheel)
from modelxlite import modelxReportConfig, modelxLiteProject


# -------------------------
# Config / Safety controls
# -------------------------
ALLOWED_BUCKETS = set(
    b.strip() for b in os.getenv("ALLOWED_BUCKETS", "").split(",") if b.strip()
)
ALLOWED_PREFIX = os.getenv("ALLOWED_PREFIX", "").strip()  # e.g. "modelxai/" (no leading slash)
REQUIRE_BEARER = os.getenv("REQUIRE_BEARER", "false").lower() == "true"
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "")


# -------------------------
# GCS helpers
# -------------------------
def parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a valid gs:// uri: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob


def enforce_allowlist(uri: str):
    bucket, blob = parse_gs_uri(uri)
    if ALLOWED_BUCKETS and bucket not in ALLOWED_BUCKETS:
        raise PermissionError(f"Bucket not allowed: {bucket}")
    if ALLOWED_PREFIX:
        # normalize
        prefix = ALLOWED_PREFIX.strip("/")
        if not blob.startswith(prefix + "/") and blob != prefix:
            raise PermissionError(f"Path not allowed (must start with {prefix}/): {blob}")


def gcs_download_to(storage_client: storage.Client, uri: str, local_path: str):
    enforce_allowlist(uri)
    bucket_name, blob_name = parse_gs_uri(uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def gcs_upload_from(storage_client: storage.Client, local_path: str, uri: str, content_type: Optional[str] = None):
    enforce_allowlist(uri)
    bucket_name, blob_name = parse_gs_uri(uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if content_type:
        blob.content_type = content_type
    blob.upload_from_filename(local_path)


def read_df_any(path: str) -> pd.DataFrame:
    # CSV or Parquet support
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    # fallback
    raise ValueError(f"Unsupported file format: {path}")


def read_series_any(path: str) -> pd.Series:
    # Expect a single column CSV or parquet
    df = read_df_any(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    # If user stored a column named target, use it
    if "target" in df.columns:
        return df["target"]
    raise ValueError(f"y file must contain 1 column (or 'target'). Got columns: {list(df.columns)}")


# -------------------------
# HTTP handler
# -------------------------
def run_modelxai(request):
    # --- Auth (optional) ---
    if REQUIRE_BEARER:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return ("Missing Bearer token", 401)
        token = auth.split(" ", 1)[1].strip()
        if not BEARER_TOKEN or token != BEARER_TOKEN:
            return ("Invalid token", 403)

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    # --- Required parameters ---
    required = [
        "model_pickle_uri",
        "X_train_uri", "y_train_uri",
        "X_test_uri", "y_test_uri",
        "output_uri",
        "run_name",
        "problem_type",
    ]
    missing = [k for k in required if k not in payload or not payload[k]]
    if missing:
        return (json.dumps({"status": "error", "error": f"Missing required fields: {missing}"}), 400, {"Content-Type": "application/json"})

    model_pickle_uri = payload["model_pickle_uri"]
    X_train_uri = payload["X_train_uri"]
    y_train_uri = payload["y_train_uri"]
    X_test_uri = payload["X_test_uri"]
    y_test_uri = payload["y_test_uri"]
    output_uri = payload["output_uri"].rstrip("/") + "/"
    run_name = payload["run_name"]
    problem_type = payload["problem_type"]  # "classification" or "regression"

    # --- Optional params ---
    bins = int(payload.get("bins", 10))
    which_reports = payload.get("which_reports", "all")  # "all" or [1,2,3,4,5]
    report_on = payload.get("report_on", "test")         # "test"|"train"|"both"
    shap_local_index = int(payload.get("shap_local_index", 0))
    pdp_features = payload.get("pdp_features", [0, 1])
    shap_sample = int(payload.get("shap_sample", 2000))

    # drift inputs (optional)
    drift_ref_uri = payload.get("drift_ref_uri", None)
    drift_cur_uri = payload.get("drift_cur_uri", None)

    # In your runner you may also accept X_train_raw/X_test_raw (optional)
    X_train_raw_uri = payload.get("X_train_raw_uri", X_train_uri)
    X_test_raw_uri = payload.get("X_test_raw_uri", X_test_uri)

    storage_client = storage.Client()

    with tempfile.TemporaryDirectory() as tmp:
        # Local file paths
        model_path = os.path.join(tmp, "model.pkl")
        Xtr_path = os.path.join(tmp, "X_train" + (".parquet" if X_train_uri.endswith(".parquet") else ".csv"))
        ytr_path = os.path.join(tmp, "y_train" + (".parquet" if y_train_uri.endswith(".parquet") else ".csv"))
        Xte_path = os.path.join(tmp, "X_test" + (".parquet" if X_test_uri.endswith(".parquet") else ".csv"))
        yte_path = os.path.join(tmp, "y_test" + (".parquet" if y_test_uri.endswith(".parquet") else ".csv"))

        Xtr_raw_path = os.path.join(tmp, "X_train_raw" + (".parquet" if X_train_raw_uri.endswith(".parquet") else ".csv"))
        Xte_raw_path = os.path.join(tmp, "X_test_raw" + (".parquet" if X_test_raw_uri.endswith(".parquet") else ".csv"))

        # Download
        gcs_download_to(storage_client, model_pickle_uri, model_path)
        gcs_download_to(storage_client, X_train_uri, Xtr_path)
        gcs_download_to(storage_client, y_train_uri, ytr_path)
        gcs_download_to(storage_client, X_test_uri, Xte_path)
        gcs_download_to(storage_client, y_test_uri, yte_path)

        gcs_download_to(storage_client, X_train_raw_uri, Xtr_raw_path)
        gcs_download_to(storage_client, X_test_raw_uri, Xte_raw_path)

        # Read
        model = joblib.load(model_path)
        X_train = read_df_any(Xtr_path)
        y_train = read_series_any(ytr_path)
        X_test = read_df_any(Xte_path)
        y_test = read_series_any(yte_path)

        X_train_raw = read_df_any(Xtr_raw_path)
        X_test_raw = read_df_any(Xte_raw_path)

        drift_ref = None
        drift_cur = None
        if drift_ref_uri and drift_cur_uri:
            dr_path = os.path.join(tmp, "drift_ref" + (".parquet" if drift_ref_uri.endswith(".parquet") else ".csv"))
            dc_path = os.path.join(tmp, "drift_cur" + (".parquet" if drift_cur_uri.endswith(".parquet") else ".csv"))
            gcs_download_to(storage_client, drift_ref_uri, dr_path)
            gcs_download_to(storage_client, drift_cur_uri, dc_path)
            drift_ref = read_df_any(dr_path)
            drift_cur = read_df_any(dc_path)

        # Run project
        cfg = modelxReportConfig(
            problem_type=problem_type,
            bins=bins,
            shap_sample=shap_sample
        )

        proj = modelxLiteProject(cfg).fit(
            model=model,
            X_train=X_train,
            y_train=y_train,
            target_name="target"
        )

        # Generate to local outputs dir
        local_out = os.path.join(tmp, "outputs")
        os.makedirs(local_out, exist_ok=True)

        saved_local = proj.generate(
            output_dir=local_out,
            run_name=run_name,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            which_reports=which_reports,
            report_on=report_on,
            shap_local_index=shap_local_index,
            pdp_features=pdp_features,
            drift_ref=drift_ref,
            drift_cur=drift_cur,
        )

        # Upload all generated html files to GCS output_uri/run_name/
        # Assume your generator writes files into: local_out/run_name/*.html
        run_dir = os.path.join(local_out, run_name)
        if not os.path.isdir(run_dir):
            # some implementations write directly to local_out; handle both
            run_dir = local_out

        uploaded = {}
        for fname in os.listdir(run_dir):
            if not fname.endswith(".html"):
                continue
            local_file = os.path.join(run_dir, fname)
            remote_uri = output_uri + run_name + "/" + fname
            gcs_upload_from(storage_client, local_file, remote_uri, content_type="text/html")
            uploaded[fname] = remote_uri

        resp = {
            "status": "success",
            "run_name": run_name,
            "uploaded": uploaded,
            "saved_local": saved_local,  # whatever your runner returns (paths)
        }
        return (json.dumps(resp), 200, {"Content-Type": "application/json"})
