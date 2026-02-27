from __future__ import annotations

import numpy as np
import pandas as pd

MSME_SCHEMA_COLUMNS = [
    "MSME_ID",
    "Sector",
    "Years_of_Operation",
    "Ownership_Type",
    "Category",
    "Location_Type",
    "Annual_Revenue",
    "Revenue_Growth_Rate",
    "Profit_Margin",
    "Debt_Outstanding",
    "Loan_to_Revenue_Ratio",
    "Number_of_Employees",
    "Capacity_Utilization",
    "Export_Percentage",
    "Technology_Level",
    "GST_Compliance_Score",
    "Inspection_Score",
    "Documentation_Readiness_Score",
    "Growth_Category",
]

SCHEME_SCHEMA_COLUMNS = [
    "Scheme_ID",
    "Scheme_Name",
    "Eligible_Sectors",
    "Max_Subsidy_Amount",
    "Target_Category",
    "Location_Criteria",
    "Impact_Factor_Revenue",
    "Impact_Factor_Employment",
]

CATEGORICAL_COLUMNS = [
    "Sector",
    "Ownership_Type",
    "Category",
    "Location_Type",
]

NUMERIC_COLUMNS = [
    "Years_of_Operation",
    "Annual_Revenue",
    "Revenue_Growth_Rate",
    "Profit_Margin",
    "Debt_Outstanding",
    "Loan_to_Revenue_Ratio",
    "Number_of_Employees",
    "Capacity_Utilization",
    "Export_Percentage",
    "Technology_Level",
    "GST_Compliance_Score",
    "Inspection_Score",
    "Documentation_Readiness_Score",
]

TARGET_COLUMN = "Growth_Category"


def create_scheme_dataset() -> pd.DataFrame:
    """Create the mandatory scheme dataset with 5 schemes."""
    schemes = [
        {
            "Scheme_ID": "SCH_01",
            "Scheme_Name": "Technology Upgrade Incentive",
            "Eligible_Sectors": "Manufacturing,Textiles,Food Processing",
            "Max_Subsidy_Amount": 2_400_000,
            "Target_Category": "Small,Medium",
            "Location_Criteria": "All",
            "Impact_Factor_Revenue": 0.18,
            "Impact_Factor_Employment": 8,
        },
        {
            "Scheme_ID": "SCH_02",
            "Scheme_Name": "Export Market Expansion Grant",
            "Eligible_Sectors": "Manufacturing,Services,IT,Textiles",
            "Max_Subsidy_Amount": 1_800_000,
            "Target_Category": "Small,Medium",
            "Location_Criteria": "Urban,Semi-Urban",
            "Impact_Factor_Revenue": 0.22,
            "Impact_Factor_Employment": 5,
        },
        {
            "Scheme_ID": "SCH_03",
            "Scheme_Name": "Rural Enterprise Support",
            "Eligible_Sectors": "Agro,Food Processing,Services",
            "Max_Subsidy_Amount": 1_100_000,
            "Target_Category": "Micro,Small",
            "Location_Criteria": "Rural,Semi-Urban",
            "Impact_Factor_Revenue": 0.12,
            "Impact_Factor_Employment": 7,
        },
        {
            "Scheme_ID": "SCH_04",
            "Scheme_Name": "Green Efficiency Subsidy",
            "Eligible_Sectors": "Manufacturing,Agro,Food Processing,Textiles",
            "Max_Subsidy_Amount": 2_600_000,
            "Target_Category": "All",
            "Location_Criteria": "All",
            "Impact_Factor_Revenue": 0.15,
            "Impact_Factor_Employment": 6,
        },
        {
            "Scheme_ID": "SCH_05",
            "Scheme_Name": "Digital Compliance Accelerator",
            "Eligible_Sectors": "IT,Services,Manufacturing,Textiles",
            "Max_Subsidy_Amount": 900_000,
            "Target_Category": "Micro,Small,Medium",
            "Location_Criteria": "Urban,Semi-Urban",
            "Impact_Factor_Revenue": 0.10,
            "Impact_Factor_Employment": 4,
        },
    ]
    return pd.DataFrame(schemes, columns=SCHEME_SCHEMA_COLUMNS)


def _sample_by_category(
    rng: np.random.Generator,
    category: str,
    value_ranges: dict[str, tuple[float, float]],
) -> float:
    low, high = value_ranges[category]
    return float(rng.uniform(low, high))


def _minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def generate_msme_dataset(n_entries: int = 360, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic MSME dataset with mandatory schema and balanced growth categories.

    The target class is created from a latent growth signal and split into terciles
    to keep class distribution balanced.
    """
    if n_entries < 300:
        raise ValueError("Dataset size must be at least 300 entries.")

    rng = np.random.default_rng(seed)

    sectors = ["Manufacturing", "Services", "Textiles", "Food Processing", "IT", "Agro"]
    ownership_types = ["Proprietorship", "Partnership", "Private Limited", "Cooperative"]
    categories = ["Micro", "Small", "Medium"]
    location_types = ["Urban", "Semi-Urban", "Rural"]

    revenue_ranges = {
        "Micro": (2_000_000, 15_000_000),
        "Small": (15_000_000, 80_000_000),
        "Medium": (80_000_000, 250_000_000),
    }
    employee_ranges = {
        "Micro": (5, 35),
        "Small": (30, 140),
        "Medium": (120, 400),
    }

    sector_growth_bias = {
        "Manufacturing": 1.0,
        "Services": 2.0,
        "Textiles": 0.5,
        "Food Processing": 1.2,
        "IT": 4.5,
        "Agro": 1.0,
    }
    sector_profit_bias = {
        "Manufacturing": 1.0,
        "Services": 2.0,
        "Textiles": 0.2,
        "Food Processing": 1.5,
        "IT": 3.0,
        "Agro": 0.8,
    }
    debt_ratio_mean = {"Micro": 0.56, "Small": 0.44, "Medium": 0.35}
    base_tech_level = {
        "Manufacturing": 3,
        "Services": 3,
        "Textiles": 2,
        "Food Processing": 3,
        "IT": 4,
        "Agro": 2,
    }

    rows: list[dict[str, object]] = []
    for idx in range(n_entries):
        category = rng.choice(categories, p=[0.48, 0.34, 0.18])
        sector = rng.choice(sectors, p=[0.22, 0.20, 0.14, 0.14, 0.15, 0.15])
        location = rng.choice(location_types, p=[0.45, 0.30, 0.25])
        ownership = rng.choice(ownership_types, p=[0.42, 0.22, 0.28, 0.08])

        years = int(rng.integers(1, 26))
        annual_revenue = _sample_by_category(rng, category, revenue_ranges)

        revenue_growth = float(
            np.clip(rng.normal(8.0 + sector_growth_bias[sector], 6.5), -8.0, 30.0)
        )
        profit_margin = float(
            np.clip(rng.normal(11.0 + sector_profit_bias[sector], 4.0), 2.0, 25.0)
        )

        debt_ratio = float(
            np.clip(rng.normal(debt_ratio_mean[category], 0.18), 0.04, 1.60)
        )
        debt_outstanding = float(annual_revenue * debt_ratio)
        loan_to_revenue_ratio = float(np.clip(debt_outstanding / annual_revenue, 0.04, 1.60))

        emp_low, emp_high = employee_ranges[category]
        employees = int(rng.integers(emp_low, emp_high + 1))

        capacity_utilization = float(np.clip(rng.normal(72, 14), 38, 98))
        export_pct = float(np.clip(rng.normal(14 if sector != "IT" else 24, 12), 0, 70))

        tech_noise = rng.normal(0, 0.9)
        tech_level = int(
            np.clip(round(base_tech_level[sector] + (0.2 if years > 10 else 0) + tech_noise), 1, 5)
        )

        gst_score = float(np.clip(rng.normal(68 + (tech_level * 4), 10), 35, 100))
        inspection_score = float(np.clip(rng.normal(66 + (years * 0.6), 11), 35, 100))
        documentation_score = float(np.clip(rng.normal(64 + (tech_level * 5), 10), 35, 100))

        rows.append(
            {
                "MSME_ID": f"MSME_{idx + 1:04d}",
                "Sector": sector,
                "Years_of_Operation": years,
                "Ownership_Type": ownership,
                "Category": category,
                "Location_Type": location,
                "Annual_Revenue": round(annual_revenue, 2),
                "Revenue_Growth_Rate": round(revenue_growth, 2),
                "Profit_Margin": round(profit_margin, 2),
                "Debt_Outstanding": round(debt_outstanding, 2),
                "Loan_to_Revenue_Ratio": round(loan_to_revenue_ratio, 3),
                "Number_of_Employees": employees,
                "Capacity_Utilization": round(capacity_utilization, 2),
                "Export_Percentage": round(export_pct, 2),
                "Technology_Level": tech_level,
                "GST_Compliance_Score": round(gst_score, 2),
                "Inspection_Score": round(inspection_score, 2),
                "Documentation_Readiness_Score": round(documentation_score, 2),
            }
        )

    df = pd.DataFrame(rows)

    latent_growth = (
        0.24 * _minmax(df["Revenue_Growth_Rate"])
        + 0.16 * _minmax(df["Profit_Margin"])
        + 0.12 * _minmax(df["Capacity_Utilization"])
        + 0.08 * _minmax(df["Export_Percentage"])
        + 0.12 * _minmax(df["Technology_Level"])
        + 0.08 * _minmax(df["GST_Compliance_Score"])
        + 0.08 * _minmax(df["Inspection_Score"])
        + 0.06 * _minmax(df["Documentation_Readiness_Score"])
        - 0.12 * _minmax(df["Loan_to_Revenue_Ratio"])
        + rng.normal(0, 0.04, size=n_entries)
    )

    growth_rank = pd.Series(latent_growth).rank(method="first")
    df[TARGET_COLUMN] = pd.qcut(growth_rank, q=3, labels=["Low", "Moderate", "High"]).astype(str)

    return df[MSME_SCHEMA_COLUMNS]

