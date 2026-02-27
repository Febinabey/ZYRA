from __future__ import annotations

import pandas as pd

CATEGORY_SUBSIDY_MULTIPLIER = {
    "Micro": 0.65,
    "Small": 0.82,
    "Medium": 1.00,
}


def _split_criteria(value: object) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def is_scheme_eligible(msme_row: pd.Series, scheme_row: pd.Series) -> bool:
    sectors = _split_criteria(scheme_row["Eligible_Sectors"])
    categories = _split_criteria(scheme_row["Target_Category"])
    locations = _split_criteria(scheme_row["Location_Criteria"])

    sector_ok = "All" in sectors or msme_row["Sector"] in sectors
    category_ok = "All" in categories or msme_row["Category"] in categories
    location_ok = "All" in locations or msme_row["Location_Type"] in locations

    return bool(sector_ok and category_ok and location_ok)


def simulate_scheme_impact(
    msme_row: pd.Series,
    scheme_row: pd.Series,
    growth_score: float,
) -> dict[str, object]:
    """Simulate one-year revenue and employment impact for an MSME-scheme pair."""
    base_revenue = float(msme_row["Annual_Revenue"])
    base_employment = int(msme_row["Number_of_Employees"])

    max_subsidy = float(scheme_row["Max_Subsidy_Amount"])
    category_multiplier = CATEGORY_SUBSIDY_MULTIPLIER.get(str(msme_row["Category"]), 0.75)

    revenue_linked_cap = (0.22 * base_revenue) + 150_000
    subsidy_amount = min(max_subsidy * category_multiplier, revenue_linked_cap)

    growth_multiplier = 0.75 + (growth_score / 200.0)
    revenue_impact_rate = float(scheme_row["Impact_Factor_Revenue"]) * growth_multiplier
    projected_revenue_increase = base_revenue * revenue_impact_rate

    projected_jobs_increase = int(
        max(
            1,
            round(float(scheme_row["Impact_Factor_Employment"]) * (0.60 + growth_score / 180.0)),
        )
    )

    return {
        "Scheme_ID": scheme_row["Scheme_ID"],
        "Scheme_Name": scheme_row["Scheme_Name"],
        "Subsidy_Amount": round(subsidy_amount, 2),
        "Base_Revenue": round(base_revenue, 2),
        "Projected_Revenue_Increase": round(projected_revenue_increase, 2),
        "Projected_Revenue_After": round(base_revenue + projected_revenue_increase, 2),
        "Base_Employment": base_employment,
        "Projected_Employment_Increase": projected_jobs_increase,
        "Projected_Employment_After": base_employment + projected_jobs_increase,
        "Revenue_Impact_Rate": round(revenue_impact_rate, 4),
    }


def enumerate_eligible_impacts(
    msme_row: pd.Series,
    schemes_df: pd.DataFrame,
    growth_score: float,
) -> list[dict[str, object]]:
    impacts: list[dict[str, object]] = []
    for _, scheme_row in schemes_df.iterrows():
        if is_scheme_eligible(msme_row, scheme_row):
            impacts.append(simulate_scheme_impact(msme_row, scheme_row, growth_score))
    return impacts


def compute_weighted_scores(
    options_df: pd.DataFrame,
    revenue_weight: float,
    employment_weight: float,
) -> pd.DataFrame:
    result_df = options_df.copy()
    if result_df.empty:
        return result_df

    weight_total = revenue_weight + employment_weight
    if weight_total <= 0:
        revenue_weight = employment_weight = 0.5
    else:
        revenue_weight = revenue_weight / weight_total
        employment_weight = employment_weight / weight_total

    max_revenue = float(result_df["Projected_Revenue_Increase"].max())
    max_employment = float(result_df["Projected_Employment_Increase"].max())

    result_df["Revenue_Normalized"] = (
        result_df["Projected_Revenue_Increase"] / max_revenue if max_revenue > 0 else 0.0
    )
    result_df["Employment_Normalized"] = (
        result_df["Projected_Employment_Increase"] / max_employment if max_employment > 0 else 0.0
    )
    result_df["Weighted_Impact_Score"] = (
        (revenue_weight * result_df["Revenue_Normalized"])
        + (employment_weight * result_df["Employment_Normalized"])
    )

    return result_df


def rank_impacts_for_advisory(
    impacts_df: pd.DataFrame,
    revenue_weight: float,
    employment_weight: float,
) -> pd.DataFrame:
    scored = compute_weighted_scores(
        impacts_df,
        revenue_weight=revenue_weight,
        employment_weight=employment_weight,
    )
    return scored.sort_values("Weighted_Impact_Score", ascending=False).reset_index(drop=True)

