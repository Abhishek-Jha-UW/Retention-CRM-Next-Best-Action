# Model & Solution — Decision Intelligence Simulator

This project is framed as **Decision Intelligence** for retention: stakeholders don’t only need churn risk—they need answers to questions like *“If we intervene this way on this cohort, what happens to revenue and downside risk?”*

---

## 1. Business problem

**Context:** Renewals teams have limited budget and capacity. Sending discounts, executive outreach, or enablement touches to everyone is expensive and sometimes backfires.

**Friction:** Classical dashboards answer *“who is risky?”* but not *“what’s the incremental value of acting, under uncertainty?”* Misaligned incentives appear when:

- High churn risk ≠ high **incremental profit** after intervention cost  
- Single-point estimates hide **tail risk** (e.g., discounts that eat margin if uptake is weak)  
- Leaders need a concise **memo** tying scenarios to assumptions, without hand-waving  

**Outcome we target:** Compare interventions on a chosen cohort using **net revenue impact distributions**, scenario **risk**, and a **transparent assumption layer**—optionally summarized by an LLM that only restates computed results.

---

## 2. What the model stack does

The solution combines four cooperating parts (engines):

### Behavioral engine

**Role:** Estimate how valuable and how “at-risk” each customer segment is **before** a new hypothetical action today.

Typical signals (tabular): tenure, usage trend, breadth of adoption, billing health, renewal window, historical support load, cohort/segment tags.

Typical outputs (examples):

- **Churn / non-renewal probability** (or survival-style probability over a horizon)  
- **Revenue at risk / expected contracted value** conditional on renewal (sometimes split as “months remaining × margin-weighted recurring revenue”—kept proportional and auditable in the app)  

Optional: text-derived **theme** features (e.g., frustration, competitor mention) via structured LLM labeling—fed as inputs, not as the final decision by itself.

### Intervention engine

**Role:** Encode *what the business can do* and how it is assumed to change behavior or economics.

Examples of interventions:

- **15% renewal discount** (recurring vs one-time—specified explicitly)  
- **High-touch save play** (account manager call + success plan)  
- **Feature / training unlock** (time-limited)  

**Critical:** “Lift” (e.g., churn reduction) is not magic. In the product we support **Tier A** sensitivity (ranges you set), and optionally **Tier B/C** when experiment logs or uplift evaluation justify tighter numbers.

### Monte Carlo simulator

**Role:** Turn point estimates into **distributions** by resampling uncertain inputs (lift, uptake, margin, baseline risk, etc.) across many runs.

Outputs include:

- Median / intervals for **net savings** after intervention cost  
- **Probability of negative net impact** (when that’s relevant)  
- Comparison between scenarios on **risk vs upside**

### LLM strategic narrator (optional)

**Role:** Produce a short **strategic memo**—recommendation, tradeoffs, risks—using **structured outputs** grounded in tables and quantiles computed in code.

**Guardrail:** Numbers in the memo must match the backend; the model does not fabricate statistics.

---

## 3. Hypothetical examples

These are **illustrative**; your app will show the same structure with your chosen data and assumptions.

### Example A — Discount vs do nothing

**Setup:** 2,000 SMB accounts renewing in 90 days. Baseline model suggests **18%** non-renewal in that window; average **margin-weighted** ARR at risk is **$4,200** per account for those flagged “at risk.”

**Intervention:** Offer a **12% price concession** for 12 months to the top 400 by expected revenue at risk. Assumed **uptake 55–75%** (uncertain), and lift reduces non-renewal by **2–6 percentage points** among takers (sensitivity band).

**Simulator output (illustrative):**

- **Median** net benefit vs no discount: **+$58k** over the horizon  
- **90%** interval: **+$12k to +$112k**  
- **Pr(net < 0)** ≈ **8%** (discount cost sometimes dominates if uptake is low or lift is weak)

**Narrator (style, not new numbers):** “Discounting is the safer bet if leadership prioritizes stability; upside is moderate but tail loss is relatively contained under stated uptake.”

### Example B — High-touch save play vs discount

**Setup:** Same cohort, but intervention B is **save play** (PM + exec sponsor + training). Higher **fixed cost per account** touched, but potentially larger lift for “strategic” accounts.

**Illustrative tradeoff:**

- Scenario B shows **higher median** upside but **wider** uncertainty and **higher** cost if execution is uneven.  
- The UI compares **median net**, **worst plausible tail** at a chosen quantile, and **resource feasibility** (# accounts touched vs capacity).

### Example C — Budget cap (“who do we touch?”)

**Setup:** \$40k marginal budget for save plays; each touch costs \$200 fully loaded.

**Simulator:** Rank accounts by expected **incremental margin** under assumed lift bands; greedy or knapsack-style selection subject to budget; Monte Carlo repeats selection noise.

**Stakeholder question answered:** “If we spend exactly the budget, what’s our **distribution** of incremental saved revenue—not just ‘expected value’ ignoring variance?”

---

## 4. How we achieve the solution (method + product)

### Problem framing

- Define **renewal horizon** and **economic unit** clearly (subscription vs contract, margin vs revenue).  
- Separate **prediction** (“who is risky?”) from **decision** (“what improves outcomes net of cost?”).  

### Behavioral modeling

- Start with **interpretable baselines** (logistic regression / GBM) with proper **train/validation** splits and **leakage checks** (no post-outcome features).  
- Report **calibration** where it matters for dollar rollups (Platt / isotonic or grouped calibration as appropriate).  
- Add **explainability** (coefficients or SHAP) for an “account brief” view—useful for trust and for the LLM’s qualitative section.

### Interventions & causality (honest tiers)

- **Tier A:** User- and literature-informed **ranges** for lift and uptake; stress tests.  
- **Tier B:** If **A/B** or rollout data exists, estimate lift on **held-out** experimental units.  
- **Tier C:** **Uplift-style** models only with **uplift metrics** (e.g., Qini/AUUC) on data that supports it—never silently equate correlation with causation.

### Uncertainty

- Monte Carlo samples **assumptions** and/or parameter posteriors—avoid “shake the point prediction” as the only noise source.  
- Present **distributions** and **probability of harm** when interventions have real cost.

### Product layer (Streamlit)

- **Pages:** Cohort builder → Scenario config → Results (distributions + comparison) → Assumptions & limitations → Memo.  
- **Reproducibility:** Seed-controlled runs; export scenario JSON for audit.

### AI integration (safely)

- **Runtime secrets** live in the hosting provider’s secret UI (Streamlit Secrets), **not** in the repository. GitHub Secrets are for **CI**, not magically for runtime unless your deploy pipeline wires them intentionally.  
- LLM prompts require **verbatim stats** injected from pandas results; optionally **structured JSON schema** validated before rendering.

---

## 5. What success looks like (portfolio signal)

Readers and interviewers should conclude:

1. You can **translate** ambiguous retention problems into measurable decisions.  
2. You respect **causal** ambiguity instead of laundering it through a leaderboard score.  
3. You quantify **risk** rather than overstating certainty.  
4. You use AI as **communication and structure**, not as a black-box oracle for money.

---

## 6. Limitations we state in-app

Models are historical; extrapolate interventions cautiously without experiments. Synthetic demos should label the **data generating process**. Any LLM-derived labels should show **evaluation** drift checks if used in scoring.
