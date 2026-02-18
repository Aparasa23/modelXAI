import requests
import json
from typing import Dict, Any


def run_modelxai_cloud(
    endpoint_url: str,
    bearer_token: str,
    payload: Dict[str, Any],
    timeout: int = 600
):
    """
    Calls the deployed ModelXAI Cloud Function.

    Parameters
    ----------
    endpoint_url : str
        https://REGION-PROJECT.cloudfunctions.net/modelxai-generate

    bearer_token : str
        Same token you configured in Cloud Function env var BEARER_TOKEN

    payload : dict
        JSON payload with gs:// paths and parameters

    timeout : int
        Seconds to wait (SHAP can take time)

    Returns
    -------
    dict : parsed JSON response
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }

    print("Submitting job to ModelXAI Cloud Function...")

    resp = requests.post(
        endpoint_url,
        headers=headers,
        json=payload,
        timeout=timeout
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Cloud Function failed [{resp.status_code}]:\n{resp.text}"
        )

    return resp.json()


if __name__ == "__main__":

    ENDPOINT = "https://REGION-PROJECT.cloudfunctions.net/modelxai-generate"
    TOKEN = "YOUR_SECRET"

    payload = {
        "run_name": "demo_run",
        "problem_type": "classification",
        "bins": 10,
        "which_reports": "all",
        "report_on": "both",

        "output_uri": "gs://your-bucket-name/modelxai/outputs/",
        "model_pickle_uri": "gs://your-bucket-name/modelxai/models/model.pkl",

        "X_train_uri": "gs://your-bucket-name/modelxai/data/X_train.csv",
        "y_train_uri": "gs://your-bucket-name/modelxai/data/y_train.csv",

        "X_test_uri": "gs://your-bucket-name/modelxai/data/X_test.csv",
        "y_test_uri": "gs://your-bucket-name/modelxai/data/y_test.csv",

        "shap_local_index": 0,
        "pdp_features": [0, 1],
        "shap_sample": 1500
    }

    result = run_modelxai_cloud(
        endpoint_url=ENDPOINT,
        bearer_token=TOKEN,
        payload=payload
    )

    print("\n=== CLOUD FUNCTION RESPONSE ===")
    print(json.dumps(result, indent=2))
