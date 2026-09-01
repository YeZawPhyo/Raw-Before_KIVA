from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(
    page_title="KIVA Microfinance Repayment Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "kiva_dashboard_artifacts"

REQUIRED_FILES = {
    "models": ARTIFACT_DIR / "optimized_models.joblib",
    "results": ARTIFACT_DIR / "optimized_results.joblib",
    "grid": ARTIFACT_DIR / "grid_search_summary.joblib",
    "importance": ARTIFACT_DIR / "permutation_importance.joblib",
    "grouped_importance": ARTIFACT_DIR / "grouped_permutation_importance.joblib",
    "consensus": ARTIFACT_DIR / "robust_consensus.joblib",
    "schema": ARTIFACT_DIR / "input_schema.joblib",
    "eda": ARTIFACT_DIR / "eda_bundle.joblib",
}

missing_files = [str(p) for p in REQUIRED_FILES.values() if not p.exists()]

if missing_files:
    st.error(
        "Model artifacts are missing. Run PART 27–30 in the notebook first, "
        "then place this app beside the kiva_dashboard_artifacts folder."
    )
    st.code("\n".join(missing_files))
    st.stop()

models = joblib.load(REQUIRED_FILES["models"])
results_df = joblib.load(REQUIRED_FILES["results"])
grid_df = joblib.load(REQUIRED_FILES["grid"])
permutation_tables = joblib.load(REQUIRED_FILES["importance"])
grouped_tables = joblib.load(REQUIRED_FILES["grouped_importance"])
consensus_df = joblib.load(REQUIRED_FILES["consensus"])
schema = joblib.load(REQUIRED_FILES["schema"])
eda = joblib.load(REQUIRED_FILES["eda"])

st.title("KIVA Microfinance Loan Repayment Prediction")
st.caption(
    "Five optimized machine-learning approaches trained with preprocessing + "
    "SMOTE inside cross-validation. Predictions are decision-support outputs, "
    "not causal conclusions."
)

tab_predict, tab_eda, tab_models, tab_importance = st.tabs(
    ["Predict", "EDA", "Model comparison", "Feature importance"]
)

with tab_predict:
    st.subheader("Loan repayment prediction")

    model_name = st.selectbox(
        "Choose a model",
        options=list(models.keys())
    )

    st.info(
        "Enter information that is known at screening time. "
        "The selected fitted pipeline automatically applies the same preprocessing "
        "used during training."
    )

    user_values = {}

    with st.form("prediction_form"):
        st.markdown("#### Loan and borrower inputs")

        columns = st.columns(2)

        for idx, feature in enumerate(schema["feature_columns"]):
            container = columns[idx % 2]

            with container:
                if feature in schema["categorical_features"]:
                    options = schema["categorical_options"].get(feature, [])
                    display_options = ["<missing>"] + options

                    selected = st.selectbox(
                        feature,
                        options=display_options,
                        key=f"cat_{feature}"
                    )

                    user_values[feature] = (
                        np.nan if selected == "<missing>" else selected
                    )

                else:
                    default = schema["numeric_defaults"].get(feature, 0.0)
                    min_value = schema["numeric_min"].get(feature, default)
                    max_value = schema["numeric_max"].get(feature, default)

                    # Number inputs are left flexible because a user may enter
                    # a legitimate future value outside the historical range.
                    user_values[feature] = st.number_input(
                        feature,
                        value=float(default),
                        key=f"num_{feature}"
                    )

        predict_clicked = st.form_submit_button(
            "Predict repayment risk",
            type="primary"
        )

    if predict_clicked:
        input_df = pd.DataFrame(
            [user_values],
            columns=schema["feature_columns"]
        )

        estimator = models[model_name]
        prediction = int(estimator.predict(input_df)[0])

        if hasattr(estimator, "predict_proba"):
            default_probability = float(estimator.predict_proba(input_df)[0, 1])
        else:
            score = float(estimator.decision_function(input_df)[0])
            default_probability = 1.0 / (1.0 + np.exp(-score))

        repayment_probability = 1.0 - default_probability

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted outcome",
            "Default risk" if prediction == 1 else "Paid / lower risk"
        )
        c2.metric(
            "Estimated default probability",
            f"{default_probability:.1%}"
        )
        c3.metric(
            "Estimated repayment probability",
            f"{repayment_probability:.1%}"
        )

        st.progress(
            min(max(default_probability, 0.0), 1.0),
            text="Predicted default probability"
        )

        st.caption(
            "This probability is produced by the selected statistical model. "
            "It should not be interpreted as certainty or as a causal judgement "
            "about an individual borrower."
        )


with tab_eda:
    st.subheader("Exploratory data analysis")

    target_counts = pd.Series(eda["target_counts"], name="Loans")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Completed-loan outcomes")
        fig, ax = plt.subplots(figsize=(6, 4))
        target_counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of unique loans")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    with c2:
        st.markdown("#### Loan amount distribution")
        loan_amount = pd.Series(eda.get("loan_amount", []), dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(loan_amount.dropna(), bins=30)
        ax.set_xlabel("Loan amount")
        ax.set_ylabel("Number of loans")
        st.pyplot(fig)

    sector_data = pd.DataFrame(eda.get("sector_summary", []))

    if not sector_data.empty:
        st.markdown("#### Default rate by sector")

        plot_sector = (
            sector_data
            .sort_values("default_rate_percent", ascending=False)
            .head(15)
            .sort_values("default_rate_percent")
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            plot_sector["sector"],
            plot_sector["default_rate_percent"]
        )
        ax.set_xlabel("Default rate (%)")
        st.pyplot(fig)

        st.caption(
            "Sector rates are descriptive associations and can be unstable for "
            "small groups. Review loan counts alongside rates."
        )

    country_data = pd.DataFrame(eda.get("country_summary", []))

    if not country_data.empty:
        st.markdown("#### Countries with at least 30 completed loans")

        plot_country = (
            country_data
            .sort_values("default_rate_percent", ascending=False)
            .head(15)
            .sort_values("default_rate_percent")
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            plot_country["location.country"],
            plot_country["default_rate_percent"]
        )
        ax.set_xlabel("Default rate (%)")
        st.pyplot(fig)

with tab_models:
    st.subheader("Optimized model comparison")

    formatted_results = results_df.copy()

    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        formatted_results[col] = formatted_results[col].map(lambda x: f"{x:.4f}")

    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )

    metric_choice = st.selectbox(
        "Metric to compare",
        ["f1", "recall", "precision", "roc_auc", "accuracy"],
        index=0
    )

    plot_df = results_df.sort_values(metric_choice)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["model"], plot_df[metric_choice])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_choice.replace("_", " ").upper())
    ax.set_title(f"Optimized model comparison — {metric_choice.upper()}")
    st.pyplot(fig)

    st.markdown("#### GridSearchCV selected parameters")

    for _, row in grid_df.iterrows():
        with st.expander(row["model"]):
            st.write("Best cross-validation F1:", round(float(row["best_cv_f1"]), 4))
            st.json(row["best_parameters"])


with tab_importance:
    st.subheader("Feature importance")

    st.warning(
        "Feature importance shows predictive association, not causation. "
        "Country, country code, currency, town, latitude and longitude overlap "
        "substantially, so their individual rankings should not be interpreted "
        "as independent effects."
    )

    importance_model = st.selectbox(
        "Choose a model for permutation importance",
        options=list(permutation_tables.keys())
    )

    table = permutation_tables[importance_model].copy()

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=min(25, len(table)),
        value=min(15, len(table))
    )

    top = table.head(top_n).sort_values("importance_mean")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"]
    )
    ax.set_xlabel("Mean decrease in test F1 after permutation")
    ax.set_title(f"Permutation importance — {importance_model}")
    st.pyplot(fig)

    st.dataframe(
        table.head(top_n),
        use_container_width=True,
        hide_index=True
    )

    grouped = grouped_tables[importance_model].copy()

    st.markdown("#### Conceptual grouped importance")

    plot_grouped = grouped.sort_values("positive_importance")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        plot_grouped["conceptual_group"],
        plot_grouped["positive_importance"]
    )
    ax.set_xlabel("Summed positive permutation importance")
    st.pyplot(fig)

    st.markdown("#### Cross-model robust consensus")
    st.dataframe(
        consensus_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Permutation importance can still be diluted when predictors are correlated. "
        "Use the grouped view and model-specific results together in the dissertation."
    )
