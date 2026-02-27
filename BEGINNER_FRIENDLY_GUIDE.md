# Beginner-Friendly Guide (Non-Technical)

This document explains the project in simple language.
It is written for someone who is not from a technical background.

---

## 1. What This Project Actually Does

Think of this platform as a decision helper for two users:

1. MSME Owner (Business Side)
- Asks: "Which government scheme is best for my business?"
- Gets:
  - growth prediction (`High/Moderate/Low`)
  - growth score (0-100)
  - eligible schemes
  - expected increase in revenue and jobs
  - best recommended scheme

2. Authority / Policy Maker (Government Side)
- Asks: "With limited subsidy budget, whom should we support first?"
- Gets:
  - optimized list of selected MSMEs
  - list of non-selected MSMEs with reasons
  - sector-wise budget distribution
  - total projected revenue and jobs generated

In short: it predicts, simulates, and allocates with transparency.

---

## 2. Simple Meaning of Important Terms

- `MSME`: Small business unit.
- `Scheme`: Subsidy/support program.
- `Growth_Category`: Predicted growth class (`Low`, `Moderate`, `High`).
- `Growth_Score`: Numeric growth potential from 0 to 100.
- `Eligible`: Business matches scheme rules (sector, category, location).
- `Projected Revenue Increase`: Estimated one-year increase in revenue.
- `Projected Employment Increase`: Estimated one-year increase in jobs.
- `Budget Constraint`: Total subsidy cannot exceed allowed budget.
- `Policy Weights`: How much importance to give:
  - revenue gain
  - employment gain

---

## 3. How to Use the App (Step-by-Step)

After running `streamlit run app.py`, use the app in this order:

1. Set scenario controls in the left sidebar:
- `Synthetic MSME entries` (keep 360 default)
- `Random Seed` (keep 42 default)
- `Total Subsidy Budget` (example: 100000000)
- `Revenue Priority Weight`
- `Employment Priority Weight`

2. Open `Layer 1: MSME Advisory Interface`
- Choose either:
  - `Use Existing MSME` (easy demo)
  - `Manual Input` (you enter your own profile)
- Check output:
  - predicted growth category
  - growth score
  - probabilities
  - recommended scheme
  - before/after table for all eligible schemes
  - explainability reasons

3. Open `Layer 2: Policy Optimization Dashboard`
- See:
  - selected MSMEs
  - non-selected MSMEs with reasons
  - sector-wise allocation
  - total projected gains
  - budget usage

4. Change policy weights and budget again
- Observe how selected MSMEs and totals change in real time.

---

## 4. Inputs You Can Give (With Cases)

Use these exact sample cases in `Manual Input` mode to see different behaviors.
Keep sidebar as default unless mentioned.

## Case A: Strong Growth MSME (Expected: High growth score, strong scheme recommendation)

Use:
- MSME_ID: `CASE_A`
- Sector: `Manufacturing`
- Years_of_Operation: `12`
- Ownership_Type: `Private Limited`
- Category: `Small`
- Location_Type: `Urban`
- Annual_Revenue: `50000000`
- Revenue_Growth_Rate: `18`
- Profit_Margin: `16`
- Debt_Outstanding: `15000000`
- Loan_to_Revenue_Ratio: `0.30`
- Number_of_Employees: `95`
- Capacity_Utilization: `84`
- Export_Percentage: `28`
- Technology_Level: `4`
- GST_Compliance_Score: `88`
- Inspection_Score: `84`
- Documentation_Readiness_Score: `86`

What you should see:
- Growth category likely `High`
- Multiple eligible schemes
- One top recommendation with strong projected gains

---

## Case B: Average MSME (Expected: Moderate results)

Use:
- MSME_ID: `CASE_B`
- Sector: `Services`
- Years_of_Operation: `7`
- Ownership_Type: `Partnership`
- Category: `Small`
- Location_Type: `Semi-Urban`
- Annual_Revenue: `26000000`
- Revenue_Growth_Rate: `8`
- Profit_Margin: `11`
- Debt_Outstanding: `11000000`
- Loan_to_Revenue_Ratio: `0.42`
- Number_of_Employees: `58`
- Capacity_Utilization: `70`
- Export_Percentage: `12`
- Technology_Level: `3`
- GST_Compliance_Score: `72`
- Inspection_Score: `70`
- Documentation_Readiness_Score: `68`

What you should see:
- Growth category often `Moderate`
- At least one eligible scheme
- Moderate projected improvements

---

## Case C: Weak Profile MSME (Expected: Lower growth score)

Use:
- MSME_ID: `CASE_C`
- Sector: `Textiles`
- Years_of_Operation: `3`
- Ownership_Type: `Proprietorship`
- Category: `Micro`
- Location_Type: `Semi-Urban`
- Annual_Revenue: `7000000`
- Revenue_Growth_Rate: `1`
- Profit_Margin: `5`
- Debt_Outstanding: `5000000`
- Loan_to_Revenue_Ratio: `0.72`
- Number_of_Employees: `18`
- Capacity_Utilization: `52`
- Export_Percentage: `4`
- Technology_Level: `2`
- GST_Compliance_Score: `56`
- Inspection_Score: `58`
- Documentation_Readiness_Score: `54`

What you should see:
- Growth category often `Low` or `Moderate`
- Scheme recommendation still appears if eligible
- Lower projected gains compared to Case A

---

## Case D: No Eligible Scheme Case (Important edge case)

Use:
- MSME_ID: `CASE_D`
- Sector: `IT`
- Years_of_Operation: `8`
- Ownership_Type: `Private Limited`
- Category: `Medium`
- Location_Type: `Rural`
- Annual_Revenue: `95000000`
- Revenue_Growth_Rate: `12`
- Profit_Margin: `14`
- Debt_Outstanding: `28000000`
- Loan_to_Revenue_Ratio: `0.29`
- Number_of_Employees: `180`
- Capacity_Utilization: `78`
- Export_Percentage: `32`
- Technology_Level: `5`
- GST_Compliance_Score: `90`
- Inspection_Score: `82`
- Documentation_Readiness_Score: `88`

What you should see:
- Prediction still works
- Advisory tab likely shows "No schemes are eligible"
- This proves system handles non-eligibility clearly

---

## Case E: Rural Employment-Focused Business

Use:
- MSME_ID: `CASE_E`
- Sector: `Agro`
- Years_of_Operation: `10`
- Ownership_Type: `Cooperative`
- Category: `Micro`
- Location_Type: `Rural`
- Annual_Revenue: `9000000`
- Revenue_Growth_Rate: `9`
- Profit_Margin: `10`
- Debt_Outstanding: `3000000`
- Loan_to_Revenue_Ratio: `0.33`
- Number_of_Employees: `26`
- Capacity_Utilization: `74`
- Export_Percentage: `5`
- Technology_Level: `3`
- GST_Compliance_Score: `75`
- Inspection_Score: `79`
- Documentation_Readiness_Score: `73`

What you should see:
- Rural-support type schemes become relevant
- If employment weight is high, this profile can become more attractive

---

## 5. Policy Dashboard Input Cases (Layer 2)

These are not manual MSME fields; these are sidebar scenario settings.

## Scenario 1: Balanced Policy
- Revenue Weight: `0.5`
- Employment Weight: `0.5`
- Budget: `100000000`

Expected behavior:
- Mix of revenue and job outcomes
- Balanced selection pattern

## Scenario 2: Revenue-First Policy
- Revenue Weight: `1.0`
- Employment Weight: `0.0`
- Budget: `100000000`

Expected behavior:
- MSMEs with higher revenue gain potential are prioritized

## Scenario 3: Employment-First Policy
- Revenue Weight: `0.0`
- Employment Weight: `1.0`
- Budget: `100000000`

Expected behavior:
- MSMEs/schemes creating more jobs are prioritized

## Scenario 4: Tight Budget
- Revenue Weight: `0.6`
- Employment Weight: `0.4`
- Budget: `15000000`

Expected behavior:
- Fewer MSMEs selected
- More non-selected MSMEs due to budget constraint

## Scenario 5: Relaxed Budget
- Revenue Weight: `0.6`
- Employment Weight: `0.4`
- Budget: `250000000`

Expected behavior:
- More MSMEs selected
- Higher total projected gains

## Scenario 6: Zero-Weight Edge Case
- Revenue Weight: `0.0`
- Employment Weight: `0.0`

Expected behavior:
- App auto-normalizes to balanced mode internally
- System still runs without failure

---

## 6. How to Read the Outputs Correctly

## In Layer 1 (Advisory)
- `Predicted Growth Category`: Class label.
- `Growth Score`: Higher means stronger growth potential.
- `Class Probabilities`: Confidence distribution across `Low/Moderate/High`.
- `Recommended Scheme`: Best under current policy weight mix.
- `Before/After`: Shows expected business improvement if scheme is applied.
- `Explainability`: Human-readable reasons for model decision.

## In Layer 2 (Policy Dashboard)
- `Selected MSMEs`: Received subsidy under optimization.
- `Non-Selected MSMEs`: Not chosen, with clear reason.
- `Sector-wise Allocation`: Which sectors got more budget.
- `Total Projected Revenue Gain`: Sum of predicted revenue improvements.
- `Total Projected Employment Generation`: Sum of predicted new jobs.
- `Budget Utilization`: How much of budget is used.
- `Budget constraint satisfied`: Must remain true.

---

## 7. What the Evaluation Phases Mean (Non-Technical)

## Phase 1: Data Design and Assumptions
What judges check:
- Did you create data in correct format?
- Are numbers realistic?
- Is growth class balanced?
- Did you clearly explain assumptions?

Simple meaning:
- "Did you build believable and well-structured data?"

## Phase 2: Growth Prediction Model
What judges check:
- Did you build a model that predicts growth category?
- Did you provide a score (0-100)?
- Did you evaluate performance?
- Can you explain why model predicts something?

Simple meaning:
- "Can your system predict growth and explain itself?"

## Phase 3: Scheme Eligibility and Impact Simulation
What judges check:
- Can one MSME be tested against multiple schemes?
- Are revenue/jobs impacts computed correctly?
- Are before/after values shown clearly?

Simple meaning:
- "Can your system show what happens if each scheme is given?"

## Phase 4: Budget-Constrained Optimization
What judges check:
- Does system choose best MSME-scheme combinations under budget?
- Are policy priorities (revenue vs jobs) actually used?
- Is budget strictly respected?
- Is selection logic explainable?

Simple meaning:
- "Can your system spend limited money smartly and transparently?"

## Phase 5: Dashboard and Integration
What judges check:
- Are both layers working in one application?
- Is simulation dynamic when sliders change?
- Are selected/non-selected outputs and reasons shown?
- Is sector summary shown?

Simple meaning:
- "Is the final product usable, transparent, and complete?"

---

## 8. Common Questions You May Face in Presentation

Q: Why synthetic data?
- Because challenge allows it and focuses on method demonstration.

Q: Why not select everyone?
- Budget is limited; optimization must prioritize impact.

Q: Why was one MSME not selected?
- Either no eligible scheme or lower priority under current weights/budget.

Q: What if policy goal changes?
- Adjust slider weights; results recompute in real time.

Q: How is fairness/transparency handled?
- Non-selected reasoning and sector-wise summaries are shown explicitly.

---

## 9. One-Line Summary for You

This project is a smart assistant that predicts MSME growth, simulates scheme impact, and helps allocate subsidy budget fairly and transparently based on policy priorities.

