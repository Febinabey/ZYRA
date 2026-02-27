from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation import compute_weighted_scores, enumerate_eligible_impacts


def build_policy_options(
    msme_df: pd.DataFrame,
    schemes_df: pd.DataFrame,
    growth_scores: pd.Series,
    revenue_weight: float,
    employment_weight: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build eligible MSME-scheme options with simulated impacts and weighted scores.
    """
    records: list[dict[str, object]] = []
    no_eligibility_ids: list[str] = []

    growth_map = growth_scores.to_dict()
    for _, msme_row in msme_df.iterrows():
        msme_id = str(msme_row["MSME_ID"])
        growth_score = float(growth_map.get(msme_id, 60.0))
        impacts = enumerate_eligible_impacts(msme_row, schemes_df, growth_score)

        if not impacts:
            no_eligibility_ids.append(msme_id)
            continue

        for impact in impacts:
            records.append(
                {
                    "MSME_ID": msme_id,
                    "Sector": msme_row["Sector"],
                    "Category": msme_row["Category"],
                    "Location_Type": msme_row["Location_Type"],
                    "Growth_Score": growth_score,
                    **impact,
                }
            )

    options_df = pd.DataFrame(records)
    if options_df.empty:
        return options_df, no_eligibility_ids

    options_df = compute_weighted_scores(
        options_df,
        revenue_weight=revenue_weight,
        employment_weight=employment_weight,
    )
    return options_df, no_eligibility_ids


def optimize_budget_allocation(
    options_df: pd.DataFrame,
    total_budget: float,
    budget_unit: int = 1_000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Solve a multiple-choice knapsack exactly using dynamic programming.

    Each MSME is a group. For each group, select at most one scheme (or none)
    while maximizing total weighted impact and respecting total budget.
    """
    if options_df.empty:
        return pd.DataFrame(), {"best_score": 0.0, "budget_used": 0.0, "budget_limit": total_budget}

    max_units = int(total_budget // budget_unit)
    if max_units <= 0:
        return pd.DataFrame(), {"best_score": 0.0, "budget_used": 0.0, "budget_limit": total_budget}

    grouped_options = [
        (msme_id, group.reset_index(drop=True))
        for msme_id, group in options_df.groupby("MSME_ID", sort=False)
    ]

    neg_inf = -1e15
    dp = np.full(max_units + 1, neg_inf, dtype=float)
    dp[0] = 0.0

    prev_unit_trackers: list[np.ndarray] = []
    pick_index_trackers: list[np.ndarray] = []
    option_sets: list[list[dict[str, object]]] = []

    for msme_id, group in grouped_options:
        option_list: list[dict[str, object]] = [
            {
                "MSME_ID": msme_id,
                "Scheme_ID": "",
                "Scheme_Name": "No Allocation",
                "Sector": group.iloc[0]["Sector"],
                "Subsidy_Amount": 0.0,
                "Projected_Revenue_Increase": 0.0,
                "Projected_Employment_Increase": 0.0,
                "Weighted_Impact_Score": 0.0,
                "Projected_Revenue_After": float(group.iloc[0]["Base_Revenue"]),
                "Projected_Employment_After": int(group.iloc[0]["Base_Employment"]),
            }
        ]

        for _, option in group.iterrows():
            option_data = option.to_dict()
            option_data["Cost_Units"] = int(np.ceil(float(option_data["Subsidy_Amount"]) / budget_unit))
            option_list.append(option_data)

        option_sets.append(option_list)

        next_dp = np.full(max_units + 1, neg_inf, dtype=float)
        prev_units = np.full(max_units + 1, -1, dtype=np.int32)
        picked_indices = np.full(max_units + 1, -1, dtype=np.int16)

        for used_units in range(max_units + 1):
            current_score = dp[used_units]
            if current_score <= neg_inf / 10:
                continue

            for option_idx, option in enumerate(option_list):
                option_units = int(option.get("Cost_Units", 0))
                next_units = used_units + option_units
                if next_units > max_units:
                    continue

                candidate_score = current_score + float(option.get("Weighted_Impact_Score", 0.0))
                if candidate_score > next_dp[next_units]:
                    next_dp[next_units] = candidate_score
                    prev_units[next_units] = used_units
                    picked_indices[next_units] = option_idx

        dp = next_dp
        prev_unit_trackers.append(prev_units)
        pick_index_trackers.append(picked_indices)

    best_units = int(np.argmax(dp))
    best_score = float(dp[best_units]) if dp[best_units] > neg_inf / 10 else 0.0

    selected_rows: list[dict[str, object]] = []
    units_cursor = best_units
    for group_idx in range(len(option_sets) - 1, -1, -1):
        option_idx = int(pick_index_trackers[group_idx][units_cursor])
        previous_units = int(prev_unit_trackers[group_idx][units_cursor])

        if option_idx < 0 or previous_units < 0:
            option_idx = 0
            previous_units = units_cursor

        chosen_option = option_sets[group_idx][option_idx]
        if chosen_option.get("Scheme_ID"):
            selected_rows.append(chosen_option)

        units_cursor = previous_units

    selected_rows.reverse()
    selected_df = pd.DataFrame(selected_rows)

    budget_used = float(selected_df["Subsidy_Amount"].sum()) if not selected_df.empty else 0.0
    summary = {
        "best_score": round(best_score, 4),
        "budget_used": round(budget_used, 2),
        "budget_limit": float(total_budget),
        "budget_utilization_pct": round((budget_used / total_budget) * 100, 2) if total_budget > 0 else 0.0,
    }
    return selected_df, summary


def build_decision_views(
    msme_df: pd.DataFrame,
    options_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    no_eligibility_ids: list[str],
    total_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    selected_ids = set(selected_df["MSME_ID"].tolist()) if not selected_df.empty else set()
    no_eligibility_set = set(no_eligibility_ids)

    if selected_df.empty:
        selected_view = pd.DataFrame(
            columns=[
                "MSME_ID",
                "Sector",
                "Scheme_ID",
                "Scheme_Name",
                "Subsidy_Amount",
                "Projected_Revenue_Increase",
                "Projected_Employment_Increase",
                "Weighted_Impact_Score",
            ]
        )
    else:
        selected_view = selected_df[
            [
                "MSME_ID",
                "Sector",
                "Scheme_ID",
                "Scheme_Name",
                "Subsidy_Amount",
                "Projected_Revenue_Increase",
                "Projected_Employment_Increase",
                "Weighted_Impact_Score",
            ]
        ].sort_values("Weighted_Impact_Score", ascending=False)

    non_selected_records: list[dict[str, object]] = []
    for _, msme_row in msme_df.iterrows():
        msme_id = msme_row["MSME_ID"]
        if msme_id in selected_ids:
            continue

        if msme_id in no_eligibility_set:
            non_selected_records.append(
                {
                    "MSME_ID": msme_id,
                    "Sector": msme_row["Sector"],
                    "Reason": "No eligible scheme for current sector/category/location criteria.",
                    "Best_Alternative_Scheme": "N/A",
                    "Required_Subsidy": 0.0,
                    "Potential_Revenue_Gain": 0.0,
                    "Potential_Employment_Gain": 0.0,
                }
            )
            continue

        msme_options = options_df[options_df["MSME_ID"] == msme_id].sort_values(
            "Weighted_Impact_Score", ascending=False
        )
        if msme_options.empty:
            non_selected_records.append(
                {
                    "MSME_ID": msme_id,
                    "Sector": msme_row["Sector"],
                    "Reason": "No eligible scheme for current sector/category/location criteria.",
                    "Best_Alternative_Scheme": "N/A",
                    "Required_Subsidy": 0.0,
                    "Potential_Revenue_Gain": 0.0,
                    "Potential_Employment_Gain": 0.0,
                }
            )
            continue

        best_option = msme_options.iloc[0]
        non_selected_records.append(
            {
                "MSME_ID": msme_id,
                "Sector": msme_row["Sector"],
                "Reason": "Not selected after global optimization under current budget and policy weights.",
                "Best_Alternative_Scheme": best_option["Scheme_Name"],
                "Required_Subsidy": float(best_option["Subsidy_Amount"]),
                "Potential_Revenue_Gain": float(best_option["Projected_Revenue_Increase"]),
                "Potential_Employment_Gain": int(best_option["Projected_Employment_Increase"]),
            }
        )

    non_selected_df = pd.DataFrame(non_selected_records)

    if selected_view.empty:
        sector_summary = pd.DataFrame(
            columns=[
                "Sector",
                "Allocated_Subsidy",
                "Selected_MSME_Count",
                "Projected_Revenue_Gain",
                "Projected_Employment_Generation",
            ]
        )
    else:
        sector_summary = (
            selected_view.groupby("Sector", as_index=False)
            .agg(
                Allocated_Subsidy=("Subsidy_Amount", "sum"),
                Selected_MSME_Count=("MSME_ID", "count"),
                Projected_Revenue_Gain=("Projected_Revenue_Increase", "sum"),
                Projected_Employment_Generation=("Projected_Employment_Increase", "sum"),
            )
            .sort_values("Allocated_Subsidy", ascending=False)
        )

    allocated = float(selected_view["Subsidy_Amount"].sum()) if not selected_view.empty else 0.0
    revenue_gain = (
        float(selected_view["Projected_Revenue_Increase"].sum()) if not selected_view.empty else 0.0
    )
    jobs_gain = (
        int(selected_view["Projected_Employment_Increase"].sum()) if not selected_view.empty else 0
    )

    totals = {
        "selected_count": len(selected_view),
        "non_selected_count": len(non_selected_df),
        "total_subsidy_allocated": round(allocated, 2),
        "total_projected_revenue_gain": round(revenue_gain, 2),
        "total_projected_employment_generation": jobs_gain,
        "budget_limit": float(total_budget),
        "budget_respected": allocated <= total_budget,
    }
    return selected_view, non_selected_df, sector_summary, totals
