import pandas as pd
import dice_ml
from dice_ml import Dice

from .html_utils import df_to_html_table, wrap_html, save_html
from .utils import to_df

def build_report3_counterfactuals(out_dir, cfg, model, train_df, target_name, X_test, query_index=0):
    sections = []

    if train_df is None:
        sections.append(("Error", "<pre class='code'>train_df is required for counterfactuals.</pre>"))
        html = wrap_html("03 – Counterfactual Explanations", sections)
        return save_html(out_dir / "03_counterfactuals.html", html)

    X = to_df(X_test)
    idx = int(max(0, min(query_index, len(X) - 1)))
    query = X.iloc[[idx]].copy()

    continuous_features = [c for c in train_df.columns
                           if c != target_name and pd.api.types.is_numeric_dtype(train_df[c])]

    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name=target_name
    )

    if cfg.problem_type == "classification":
        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")

        dice_r = Dice(data, m, method="random")
        cf_r = dice_r.generate_counterfactuals(query_instances=query, total_CFs=3, desired_class="opposite")
        cf_r_df = cf_r.cf_examples_list[0].final_cfs_df
        sections.append(("Random Sampling Counterfactuals", df_to_html_table(cf_r_df)))

        dice_g = Dice(data, m, method="genetic")
        cf_g = dice_g.generate_counterfactuals(query_instances=query, total_CFs=3, desired_class="opposite")
        cf_g_df = cf_g.cf_examples_list[0].final_cfs_df
        sections.append(("Genetic / Evolutionary Counterfactuals", df_to_html_table(cf_g_df)))

        sections.append(("Notes", "<pre class='code'>Counterfactual-based global importance: aggregate absolute feature deltas across CFs and rank.</pre>"))

    else:
        m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
        dice_r = Dice(data, m, method="random")

        pred0 = float(model.predict(query)[0])
        desired_range = [pred0 * 0.9, pred0 * 1.1]

        cf_r = dice_r.generate_counterfactuals(query_instances=query, total_CFs=3, desired_range=desired_range)
        cf_r_df = cf_r.cf_examples_list[0].final_cfs_df
        sections.append(("Random Sampling Counterfactuals (Regression)", df_to_html_table(cf_r_df)))
        sections.append(("Notes", f"<pre class='code'>Desired range used: {desired_range}</pre>"))

    html = wrap_html("03 – Counterfactual Explanations", sections)
    return save_html(out_dir / "03_counterfactuals.html", html)
