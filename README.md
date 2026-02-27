# AI-Powered Dual-Layer MSME Growth Advisory & Subsidy Optimization Platform

This project implements the full software-only solution for:
- MSME-level growth prediction and scheme advisory
- Authority-level budget-constrained subsidy optimization

It follows the provided hackathon schema and constraints, including synthetic data generation, single-year impact simulation, and transparent selection logic.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Covers

### Layer 1: MSME Advisory Interface
- Accepts MSME profile input (existing synthetic record or manual input).
- Predicts `Growth_Category` (`High / Moderate / Low`) using ML.
- Converts prediction into a `Growth Score` (0-100).
- Identifies all eligible schemes.
- Simulates one-year:
  - Revenue increase
  - Employment increase
- Recommends the best scheme based on weighted impact.
- Shows before/after projections and explainability.

### Layer 2: Policy Optimization Dashboard
- Aggregates all MSMEs.
- Accepts budget constraint.
- Accepts adjustable policy sliders:
  - Revenue priority weight
  - Employment priority weight
- Computes weighted impact scores.
- Optimizes MSME-scheme selection with exact multiple-choice knapsack DP such that:
  - `Total Subsidy Allocated <= Budget`
  - Each MSME gets at most one scheme
- Displays:
  - Selected MSMEs
  - Non-selected MSMEs with justification
  - Sector-wise allocation distribution
  - Total projected revenue gain
  - Total projected employment generation

## Dataset Schema

The generated datasets follow the mandatory fields exactly:
- MSME dataset includes all required profile, financial, operational, compliance, and target fields.
- Scheme dataset includes 5 schemes with all required eligibility and impact columns.

See `ASSUMPTIONS.md` for documented assumptions and ranges.

## Technical Design

- `msme_platform/data.py`: synthetic dataset generation + schema constants
- `msme_platform/model.py`: ML training, evaluation, growth scoring, explainability helpers
- `msme_platform/simulation.py`: eligibility logic + impact simulation
- `msme_platform/optimization.py`: weighted scoring + budget-constrained optimization + transparent decision views
- `app.py`: integrated Streamlit UI for both layers

## Constraint Compliance

- Software-only implementation
- Synthetic dataset
- Minimum 300 MSMEs supported
- Maximum 5 schemes used
- Single-year projection
- One scheme per MSME in optimization
- Strict budget enforcement in optimizer
