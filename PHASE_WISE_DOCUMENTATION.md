# Phase-Wise Documentation

## Problem Statement
AI-Powered Dual-Layer MSME Growth Advisory & Subsidy Optimization Platform

## Solution Scope
This implementation delivers a software-only, data-driven dual-layer platform:
- Layer 1: MSME advisory with growth prediction, growth score, scheme eligibility, impact simulation, recommendation, and explainability.
- Layer 2: Policy dashboard with budget-constrained optimization, revenue-employment trade-off controls, and transparent selection/non-selection outputs.

---

## Phase 1: Data Design and Assumptions (15 Marks)

### Checkpoint Coverage
- Mandatory schema is implemented for MSME and scheme datasets.
- Synthetic MSME data generation supports minimum 300 entries (default 360).
- Value ranges are controlled and clipped to realistic business limits.
- `Growth_Category` is balanced using a latent-growth signal split into terciles (`Low`, `Moderate`, `High`).
- Assumptions are documented in [ASSUMPTIONS.md](/c:/Users/febin/ZYRA/ASSUMPTIONS.md).

### Where It Is Implemented
- Data schema and synthetic generation: [data.py](/c:/Users/febin/ZYRA/msme_platform/data.py)
- Assumptions: [ASSUMPTIONS.md](/c:/Users/febin/ZYRA/ASSUMPTIONS.md)

### Evaluation Mapping
- Data realism: achieved through category-wise financial/employee ranges, sector biases, bounded scores.
- Structural adherence: all mandatory fields are present in both datasets.
- Assumption clarity: all major modeling/data assumptions are explicitly documented.

---

## Phase 2: Growth Prediction Model (20 Marks)

### Checkpoint Coverage
- Machine learning model implemented using `RandomForestClassifier`.
- Predicts `Growth_Category` (`High / Moderate / Low`).
- Converts probabilities into `Growth_Score` (0-100) using weighted class expectation.
- Reports evaluation metrics: accuracy and macro-F1, plus full classification report.
- Explainability included via:
  - Global feature importance aggregation
  - Local reasoning rules for profile-level interpretation

### Core Logic
- Train/test split with stratification.
- Mixed preprocessing pipeline:
  - One-hot encoding for categorical features
  - Standard scaling for numeric features
- Growth score formula:
  - `Growth_Score = Sum(P(class_i) * Weight(class_i))`
  - Weights used: `Low=25`, `Moderate=60`, `High=95`

### Where It Is Implemented
- Model training, scoring, evaluation, explainability: [model.py](/c:/Users/febin/ZYRA/msme_platform/model.py)
- Display in UI: [app.py](/c:/Users/febin/ZYRA/app.py)

### Evaluation Mapping
- Model design and logic: full ML pipeline with robust preprocessing.
- Evaluation and validation: quantitative metrics exposed in dashboard.
- Explainability: both feature-importance and profile-level reason generation.

---

## Phase 3: Scheme Eligibility and Impact Simulation (20 Marks)

### Checkpoint Coverage
- Multi-scheme eligibility based on sector, target category, and location criteria.
- Revenue impact simulation for each eligible scheme.
- Employment impact simulation for each eligible scheme.
- Before/after projections shown for revenue and employment.
- Mathematical formulas are implemented deterministically for reproducible results.

### Core Logic
- Eligibility:
  - Sector match OR `All`
  - Category match OR `All`
  - Location match OR `All`
- Subsidy computation:
  - `Subsidy = min(Max_Subsidy * Category_Multiplier, 0.22 * Annual_Revenue + 150000)`
- Revenue increase:
  - `Projected_Revenue_Increase = Annual_Revenue * Impact_Factor_Revenue * (0.75 + Growth_Score/200)`
- Employment increase:
  - `Projected_Employment_Increase = max(1, round(Impact_Factor_Employment * (0.60 + Growth_Score/180)))`

### Where It Is Implemented
- Eligibility + simulation formulas: [simulation.py](/c:/Users/febin/ZYRA/msme_platform/simulation.py)
- Advisory table display: [app.py](/c:/Users/febin/ZYRA/app.py)

### Evaluation Mapping
- Simulation correctness: formula-based and constrained simulation.
- Multi-scheme handling: all eligible schemes are evaluated and ranked.
- Projection clarity: base vs projected metrics displayed in advisory UI.

---

## Phase 4: Budget-Constrained Optimization Engine (25 Marks)

### Checkpoint Coverage
- Optimization logic implemented using exact dynamic programming (multiple-choice knapsack).
- Revenue-employment weighted scoring integrated.
- Policy sliders integrated in UI and fed into optimizer.
- Budget constraint strictly enforced (`Total Allocated <= Budget`).
- Logical ranking and selection performed globally over MSME-scheme options.

### Core Logic
- Weighted impact score:
  - Revenue and employment impacts are normalized across options.
  - `Weighted_Impact_Score = w_r * Revenue_Normalized + w_e * Employment_Normalized`
- Constraint:
  - Select at most one scheme per MSME.
  - Maximize total weighted impact under total budget cap.

### Where It Is Implemented
- Option generation + optimization: [optimization.py](/c:/Users/febin/ZYRA/msme_platform/optimization.py)
- Weight controls and output dashboards: [app.py](/c:/Users/febin/ZYRA/app.py)

### Evaluation Mapping
- Optimization correctness: exact DP (not greedy heuristic), supports one-scheme-per-MSME condition.
- Policy weight integration: real-time reruns with slider updates.
- Decision reasoning: explicit non-selection reasons and best alternative scheme shown.

---

## Phase 5: Dashboard, Transparency and System Integration (20 Marks)

### Checkpoint Coverage
- Functional MSME advisory interface (Layer 1).
- Functional policy optimization dashboard (Layer 2).
- Real-time simulation with weight and budget changes.
- Selected MSMEs displayed.
- Non-selected MSMEs displayed with reasons.
- Sector-wise summary visualized.

### Dashboard Outputs
- Selected MSME list with chosen scheme, subsidy, projected gains.
- Non-selected MSME list with justification:
  - No eligibility, or
  - Not selected under current global optimization and budget.
- Sector-wise subsidy distribution and projected outcomes.
- Total projected revenue gain and employment generation.
- Budget utilization and budget compliance status.

### Where It Is Implemented
- End-to-end UI integration: [app.py](/c:/Users/febin/ZYRA/app.py)
- Supporting core modules: [msme_platform](/c:/Users/febin/ZYRA/msme_platform)

### Evaluation Mapping
- Interface usability: tab-wise separation and structured metrics.
- Transparency: reasons and alternatives included in non-selected view.
- Policy simulation: slider-driven re-optimization.
- Integration: all modules connected in one executable app.

---

## Constraint Compliance Summary
- Software-only: Yes
- Synthetic data: Yes
- Single-year projection: Yes
- Maximum 5 schemes: Yes
- One scheme per MSME: Yes (enforced in optimization)
- Strict budget enforcement: Yes

---

## Expected Outcome Alignment
This solution demonstrates:
- Predictive growth modeling
- Multi-scheme impact simulation
- Budget-based optimization
- Revenue-employment trade-off analysis
- Transparent and explainable decision-making

