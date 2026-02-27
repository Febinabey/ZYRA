from __future__ import annotations

import pandas as pd
import streamlit as st

from msme_platform.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    create_scheme_dataset,
    generate_msme_dataset,
)
from msme_platform.model import (
    generate_local_reasons,
    predict_growth,
    score_msme_dataset,
    train_growth_model,
)
from msme_platform.optimization import (
    build_decision_views,
    build_policy_options,
    optimize_budget_allocation,
)
from msme_platform.simulation import rank_impacts_for_advisory


def _format_currency(value: float) -> str:
    return f"{value:,.0f}"


def _normalize_weights(revenue_weight: float, employment_weight: float) -> tuple[float, float]:
    total = revenue_weight + employment_weight
    if total <= 0:
        return 0.5, 0.5
    return revenue_weight / total, employment_weight / total


def _build_manual_profile(default_row: pd.Series, sectors: list[str]) -> dict[str, object]:
    col1, col2, col3 = st.columns(3)
    with col1:
        msme_id = st.text_input("MSME_ID", value="CUSTOM_0001")
        sector = st.selectbox("Sector", options=sectors, index=0)
        years = st.number_input(
            "Years_of_Operation",
            min_value=1,
            max_value=40,
            value=int(default_row["Years_of_Operation"]),
        )
        ownership = st.selectbox(
            "Ownership_Type",
            options=["Proprietorship", "Partnership", "Private Limited", "Cooperative"],
            index=0,
        )
        category = st.selectbox("Category", options=["Micro", "Small", "Medium"], index=0)
        location = st.selectbox("Location_Type", options=["Urban", "Semi-Urban", "Rural"], index=0)

    with col2:
        annual_revenue = st.number_input(
            "Annual_Revenue",
            min_value=100_000.0,
            value=float(default_row["Annual_Revenue"]),
            step=100_000.0,
        )
        revenue_growth = st.number_input(
            "Revenue_Growth_Rate",
            min_value=-20.0,
            max_value=40.0,
            value=float(default_row["Revenue_Growth_Rate"]),
        )
        profit_margin = st.number_input(
            "Profit_Margin",
            min_value=0.0,
            max_value=40.0,
            value=float(default_row["Profit_Margin"]),
        )
        debt_outstanding = st.number_input(
            "Debt_Outstanding",
            min_value=0.0,
            value=float(default_row["Debt_Outstanding"]),
            step=100_000.0,
        )
        loan_to_revenue = st.number_input(
            "Loan_to_Revenue_Ratio",
            min_value=0.0,
            max_value=2.5,
            value=float(default_row["Loan_to_Revenue_Ratio"]),
            step=0.01,
        )

    with col3:
        employees = st.number_input(
            "Number_of_Employees",
            min_value=1,
            max_value=1000,
            value=int(default_row["Number_of_Employees"]),
        )
        capacity = st.number_input(
            "Capacity_Utilization",
            min_value=0.0,
            max_value=100.0,
            value=float(default_row["Capacity_Utilization"]),
        )
        export_pct = st.number_input(
            "Export_Percentage",
            min_value=0.0,
            max_value=100.0,
            value=float(default_row["Export_Percentage"]),
        )
        tech_level = st.number_input(
            "Technology_Level",
            min_value=1,
            max_value=5,
            value=int(default_row["Technology_Level"]),
        )
        gst_score = st.number_input(
            "GST_Compliance_Score",
            min_value=0.0,
            max_value=100.0,
            value=float(default_row["GST_Compliance_Score"]),
        )
        inspection_score = st.number_input(
            "Inspection_Score",
            min_value=0.0,
            max_value=100.0,
            value=float(default_row["Inspection_Score"]),
        )
        docs_score = st.number_input(
            "Documentation_Readiness_Score",
            min_value=0.0,
            max_value=100.0,
            value=float(default_row["Documentation_Readiness_Score"]),
        )

    return {
        "MSME_ID": msme_id,
        "Sector": sector,
        "Years_of_Operation": int(years),
        "Ownership_Type": ownership,
        "Category": category,
        "Location_Type": location,
        "Annual_Revenue": float(annual_revenue),
        "Revenue_Growth_Rate": float(revenue_growth),
        "Profit_Margin": float(profit_margin),
        "Debt_Outstanding": float(debt_outstanding),
        "Loan_to_Revenue_Ratio": float(loan_to_revenue),
        "Number_of_Employees": int(employees),
        "Capacity_Utilization": float(capacity),
        "Export_Percentage": float(export_pct),
        "Technology_Level": int(tech_level),
        "GST_Compliance_Score": float(gst_score),
        "Inspection_Score": float(inspection_score),
        "Documentation_Readiness_Score": float(docs_score),
    }


def _select_profile(msme_df: pd.DataFrame) -> pd.DataFrame | None:
    st.subheader("MSME Profile Input")
    mode = st.radio(
        "Input Mode",
        options=["Use Existing MSME", "Manual Input"],
        horizontal=True,
    )

    if mode == "Use Existing MSME":
        selected_id = st.selectbox("Select MSME_ID", options=msme_df["MSME_ID"].tolist())
        selected_row = msme_df[msme_df["MSME_ID"] == selected_id].iloc[0]
        return pd.DataFrame([selected_row.drop(labels=["Growth_Category"])])

    default_row = msme_df.iloc[0]
    with st.form("manual_profile_form"):
        manual_profile = _build_manual_profile(default_row, sorted(msme_df["Sector"].unique()))
        submitted = st.form_submit_button("Run Advisory")
    if submitted:
        return pd.DataFrame([manual_profile])

    return None


def main() -> None:
    st.set_page_config(
        page_title="MSME Dual-Layer Advisory & Subsidy Optimizer",
        layout="wide",
    )
    st.title("AI-Powered Dual-Layer MSME Growth Advisory & Subsidy Optimization Platform")
    st.caption("Software-only prototype using synthetic data and single-year impact simulation.")

    with st.sidebar:
        st.header("Scenario Controls")
        n_entries = st.slider("Synthetic MSME entries", min_value=300, max_value=1200, value=360, step=20)
        random_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
        total_budget = st.number_input(
            "Total Subsidy Budget",
            min_value=1_000_000.0,
            value=100_000_000.0,
            step=1_000_000.0,
        )
        raw_revenue_weight = st.slider(
            "Revenue Priority Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
        raw_employment_weight = st.slider(
            "Employment Priority Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
        )
        revenue_weight, employment_weight = _normalize_weights(
            raw_revenue_weight, raw_employment_weight
        )
        st.write(
            f"Normalized policy mix -> Revenue: {revenue_weight:.2f}, Employment: {employment_weight:.2f}"
        )

    msme_df = generate_msme_dataset(n_entries=n_entries, seed=int(random_seed))
    schemes_df = create_scheme_dataset()
    model_artifacts = train_growth_model(msme_df, random_state=int(random_seed))
    scored_msme_df = score_msme_dataset(model_artifacts, msme_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSMEs", len(msme_df))
    col2.metric("Schemes", len(schemes_df))
    col3.metric("Model Accuracy", f"{model_artifacts.metrics['accuracy']:.3f}")
    col4.metric("Macro F1", f"{model_artifacts.metrics['macro_f1']:.3f}")

    tab1, tab2, tab3 = st.tabs(
        [
            "Layer 1: MSME Advisory Interface",
            "Layer 2: Policy Optimization Dashboard",
            "Data, Metrics & Transparency",
        ]
    )

    with tab1:
        profile_df = _select_profile(msme_df)
        if profile_df is None:
            st.info("Submit a manual profile or select an existing MSME to run advisory.")
        else:
            prediction = predict_growth(model_artifacts, profile_df)
            profile_row = profile_df.iloc[0]

            pred_col1, pred_col2 = st.columns(2)
            pred_col1.metric("Predicted Growth Category", prediction["predicted_category"])
            pred_col2.metric("Growth Score (0-100)", f"{prediction['growth_score']:.1f}")

            probability_df = pd.DataFrame([prediction["probabilities"]])
            st.write("Class Probabilities")
            st.dataframe(probability_df.style.format("{:.2%}"), use_container_width=True)

            options_df, _ = build_policy_options(
                msme_df=profile_df.assign(Growth_Category="Unknown"),
                schemes_df=schemes_df,
                growth_scores=pd.Series(
                    [prediction["growth_score"]], index=[profile_row["MSME_ID"]]
                ),
                revenue_weight=revenue_weight,
                employment_weight=employment_weight,
            )

            if options_df.empty:
                st.warning("No schemes are eligible for this MSME profile.")
            else:
                ranked_options = rank_impacts_for_advisory(
                    impacts_df=options_df,
                    revenue_weight=revenue_weight,
                    employment_weight=employment_weight,
                )
                best = ranked_options.iloc[0]
                st.subheader(
                    f"Recommended Scheme: {best['Scheme_Name']} ({best['Scheme_ID']})"
                )
                st.write(
                    f"Reason: highest weighted one-year impact under current policy mix "
                    f"(Revenue {revenue_weight:.2f}, Employment {employment_weight:.2f})."
                )

                display_cols = [
                    "Scheme_ID",
                    "Scheme_Name",
                    "Subsidy_Amount",
                    "Base_Revenue",
                    "Projected_Revenue_Increase",
                    "Projected_Revenue_After",
                    "Base_Employment",
                    "Projected_Employment_Increase",
                    "Projected_Employment_After",
                    "Weighted_Impact_Score",
                ]
                st.write("Before vs After Projection (All Eligible Schemes)")
                st.dataframe(
                    ranked_options[display_cols].sort_values("Weighted_Impact_Score", ascending=False),
                    use_container_width=True,
                )

            st.write("Prediction Explainability")
            reasons = generate_local_reasons(profile_row, model_artifacts.reference_medians)
            for reason in reasons:
                st.write(f"- {reason}")

            importance_df = model_artifacts.feature_importance.head(10).set_index("feature")
            st.write("Top Global Feature Importance")
            st.bar_chart(importance_df["importance_pct"])

    with tab2:
        growth_scores = scored_msme_df.set_index("MSME_ID")["Growth_Score"]
        policy_options_df, no_eligibility_ids = build_policy_options(
            msme_df=msme_df,
            schemes_df=schemes_df,
            growth_scores=growth_scores,
            revenue_weight=revenue_weight,
            employment_weight=employment_weight,
        )
        selected_df, optimization_summary = optimize_budget_allocation(
            options_df=policy_options_df,
            total_budget=float(total_budget),
        )
        selected_view, non_selected_df, sector_summary_df, totals = build_decision_views(
            msme_df=msme_df,
            options_df=policy_options_df,
            selected_df=selected_df,
            no_eligibility_ids=no_eligibility_ids,
            total_budget=float(total_budget),
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Selected MSMEs", int(totals["selected_count"]))
        m2.metric("Subsidy Allocated", _format_currency(totals["total_subsidy_allocated"]))
        m3.metric("Revenue Gain (Projected)", _format_currency(totals["total_projected_revenue_gain"]))
        m4.metric(
            "Employment Generation (Projected)",
            int(totals["total_projected_employment_generation"]),
        )
        m5.metric("Budget Utilization", f"{optimization_summary['budget_utilization_pct']:.1f}%")

        if totals["budget_respected"]:
            st.success("Budget constraint satisfied.")
        else:
            st.error("Budget constraint violated.")

        st.subheader("Selected MSMEs")
        st.dataframe(selected_view, use_container_width=True)

        st.subheader("Non-Selected MSMEs with Justification")
        st.dataframe(non_selected_df, use_container_width=True)

        st.subheader("Sector-Wise Allocation Distribution")
        if sector_summary_df.empty:
            st.info("No sector allocation available under current constraints.")
        else:
            st.dataframe(sector_summary_df, use_container_width=True)
            st.bar_chart(sector_summary_df.set_index("Sector")["Allocated_Subsidy"])

    with tab3:
        st.subheader("Mandatory Dataset Snapshot")
        st.write("MSME Dataset (sample)")
        st.dataframe(msme_df.head(20), use_container_width=True)

        st.write("Scheme Dataset")
        st.dataframe(schemes_df, use_container_width=True)

        st.subheader("Model Evaluation")
        st.dataframe(
            model_artifacts.classification_report[
                ["label", "precision", "recall", "f1-score", "support"]
            ],
            use_container_width=True,
        )

        st.subheader("Growth Category Distribution")
        growth_counts = msme_df["Growth_Category"].value_counts().sort_index()
        st.bar_chart(growth_counts)

        st.subheader("Schema Check")
        st.write(
            "Feature columns used by model:",
            CATEGORICAL_COLUMNS + NUMERIC_COLUMNS,
        )
        st.write("All outputs include reasoning and allocation transparency fields.")


if __name__ == "__main__":
    main()
