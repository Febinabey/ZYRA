# Assumptions Used for Synthetic Data and Simulation

## Data Generation
- Dataset size is configurable, with default `360` MSMEs and minimum `300`.
- Sector mix is weighted toward manufacturing and services to reflect broad MSME participation.
- Category mix defaults to `Micro > Small > Medium`.
- Revenue ranges (INR):
  - `Micro`: 2,000,000 to 15,000,000
  - `Small`: 15,000,000 to 80,000,000
  - `Medium`: 80,000,000 to 250,000,000
- Employment ranges:
  - `Micro`: 5 to 35
  - `Small`: 30 to 140
  - `Medium`: 120 to 400
- Compliance and operational scores are clipped to realistic 0-100 bounds.
- `Growth_Category` is assigned from a latent growth signal and split into terciles for balanced classes.

## Scheme Design
- Exactly 5 schemes are used (maximum allowed).
- Eligibility checks are based on sector, category, and location matching.
- `Impact_Factor_Revenue` values are within 5%-25%.
- `Impact_Factor_Employment` values are within 1-10 jobs.

## Impact Simulation
- Single-year projection only.
- Scheme subsidy is capped by:
  - Scheme max amount
  - MSME category multiplier (`Micro < Small < Medium`)
  - Revenue-linked cap (`22% of annual revenue + fixed allowance`)
- Projected revenue increase is:
  - `Annual_Revenue * Impact_Factor_Revenue * Growth_Adjustment`
- Projected employment increase scales from scheme employment factor and growth score.

## Policy Optimization
- Each MSME can receive at most one scheme.
- Selection uses exact multiple-choice knapsack dynamic programming.
- Objective maximizes weighted impact score:
  - Revenue and employment impacts are normalized and weighted by policy sliders.
- Total allocated subsidy is enforced to stay within budget.

