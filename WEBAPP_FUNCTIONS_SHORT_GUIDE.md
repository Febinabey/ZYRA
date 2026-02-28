# Webapp Functions Short Guide (Beginner Friendly)

This guide explains what each part of your webapp does in simple language.

## 1) What this webapp does
- Predicts whether an MSME has `High`, `Moderate`, or `Low` growth potential.
- Finds which schemes an MSME is eligible for.
- Simulates expected revenue and job increase for each eligible scheme.
- Chooses MSMEs and schemes under a fixed budget using optimization.
- Shows transparent reasons for selected and non-selected MSMEs.

## 2) Full flow in simple steps
1. Generate synthetic MSME + scheme data.
2. Train ML model and score MSMEs.
3. Simulate impacts for eligible schemes.
4. Optimize subsidy allocation under budget.
5. Show advisory, policy, and transparency dashboards.

## 3) File-wise explanation

| File | What it does |
|---|---|
| `app.py` | Main Streamlit webapp (UI + user flow + charts + scenario tools). |
| `msme_platform/data.py` | Builds synthetic MSME and scheme datasets with required schema. |
| `msme_platform/model.py` | Trains ML model, predicts growth, computes growth score, creates explanations. |
| `msme_platform/simulation.py` | Checks eligibility and simulates one-year revenue/jobs impact. |
| `msme_platform/optimization.py` | Runs budget-constrained optimization and prepares selected/non-selected outputs. |

## 4) `app.py` function explanations

| Function | Short explanation |
|---|---|
| `_get_synthetic_data` | Creates MSME and scheme data once and caches it for speed. |
| `_get_model_bundle` | Trains model + scores MSMEs, then caches results. |
| `_run_policy_engine` | Runs option building, optimization, and decision views in one pipeline. |
| `_inject_theme` | Applies custom dark dashboard styling (colors, spacing, cards). |
| `_format_currency` | Converts numbers to readable currency format (comma-separated). |
| `_normalize_weights` | Makes revenue/employment weights sum to 1. |
| `_render_topbar` | Draws top header bar with app title and tagline. |
| `_render_section` | Draws section title, subtitle, and divider line. |
| `_growth_badge` | Shows color badge for `High`, `Moderate`, or `Low` growth. |
| `_kpi_card` | Reusable metric card UI component. |
| `_advisory_card` | Reusable recommendation card UI component. |
| `_optimization_summary_card` | Shows budget status text + budget utilization gauge/progress. |
| `_styled_table` | Applies dark striped style to tables for readability. |
| `_build_manual_profile` | Collects manual MSME inputs from the form. |
| `_select_profile` | Lets user pick existing MSME or submit manual profile. |
| `_plot_probability` | Plots growth class probabilities (High/Moderate/Low). |
| `_plot_sector_allocation` | Plots subsidy allocation by sector. |
| `_plot_feature_importance` | Plots top model features affecting predictions. |
| `_plot_donut` | Reusable donut chart for splits (selection, sector mix, etc.). |
| `_build_summary_from_selected` | Recomputes summary totals from selected rows. |
| `_apply_rural_guardrail` | Enforces minimum rural share by replacing low-priority non-rural picks where possible. |
| `main` | Main app runner: sidebar controls, caching pipeline, tabs, outputs. |

## 5) `msme_platform/data.py` functions

| Function | Short explanation |
|---|---|
| `create_scheme_dataset` | Creates fixed scheme table (5 schemes) with eligibility and impact factors. |
| `_sample_by_category` | Helper: samples value range based on MSME category. |
| `_minmax` | Helper: normalizes values to 0-1 for scoring. |
| `generate_msme_dataset` | Generates synthetic MSME dataset (minimum 300 rows) with balanced growth categories. |

## 6) `msme_platform/model.py` functions

| Function | Short explanation |
|---|---|
| `_base_feature_name` | Helper: converts encoded feature names to base names. |
| `_aggregate_feature_importance` | Helper: combines feature importances into readable table. |
| `train_growth_model` | Trains RandomForest model, returns metrics/report/feature importance. |
| `predict_growth` | Predicts class and growth score for one MSME profile. |
| `score_msme_dataset` | Scores all MSMEs with predicted class and probabilities. |
| `generate_local_reasons` | Creates human-readable reasons for prediction behavior. |

## 7) `msme_platform/simulation.py` functions

| Function | Short explanation |
|---|---|
| `_split_criteria` | Helper: splits comma-separated criteria values. |
| `is_scheme_eligible` | Checks if MSME matches scheme sector/category/location rules. |
| `simulate_scheme_impact` | Calculates one-year subsidy, revenue increase, and job increase for one MSME-scheme pair. |
| `enumerate_eligible_impacts` | Runs simulation for all eligible schemes of one MSME. |
| `compute_weighted_scores` | Creates weighted impact score using revenue and employment priorities. |
| `rank_impacts_for_advisory` | Sorts scheme options by weighted impact score for recommendation. |

## 8) `msme_platform/optimization.py` functions

| Function | Short explanation |
|---|---|
| `build_policy_options` | Builds all MSME-scheme options with simulated impacts and scores. |
| `optimize_budget_allocation` | Uses dynamic programming knapsack to maximize impact under budget. |
| `build_decision_views` | Creates final tables: selected, non-selected (with reasons), sector summary, totals. |

## 9) Main outputs you see in UI
- Predicted growth category and growth score.
- Recommended scheme with before/after projections.
- Selected MSMEs under budget.
- Non-selected MSMEs with reasons.
- Sector allocation charts.
- ROI, cost-per-job, scenario compare, and fairness guardrail effects.

## 10) Simple terms

| Term | Meaning |
|---|---|
| MSME | A small/medium business record in the dataset. |
| Growth Category | Predicted class: Low, Moderate, High. |
| Growth Score | Numeric growth potential score (0-100). |
| Eligible Scheme | Scheme that matches MSME rules. |
| Projection | Expected one-year outcome after subsidy. |
| Optimization | Best selection under budget and policy priorities. |
| ROI | Revenue gain divided by subsidy allocated. |
| Cost per Job | Subsidy allocated per projected job created. |
| Guardrail | Extra policy rule, e.g., minimum rural selection share. |

