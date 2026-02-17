from modelxlite import modelxReportConfig, modelxLiteProject

# Example usage (binary classification):
# model = ... trained sklearn classifier with predict_proba
# X_train, y_train, X_test, y_test should be ready

cfg = modelxReportConfig(problem_type="classification", bins=10)

fx = modelxLiteProject(cfg).fit(
    model=model,
    X_train=X_train,
    y_train=y_train,
    target_name="target"
)

saved = fx.generate(
    output_dir="outputs",
    run_name="demo_run",
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    X_train_raw=X_train,   # optional for feature means in bin tables
    X_test_raw=X_test,
    which_reports="all",    # or [1,2,3,4]
    report_on="both",       # "test" | "train" | "both"
    shap_local_index=0,
    pdp_features=[0, 1],
    drift_ref=X_train,
    drift_cur=X_test,
)

print(saved)
