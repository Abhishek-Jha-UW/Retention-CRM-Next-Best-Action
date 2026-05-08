# Decision Intelligence Simulator — Project Flow

This document is the **logical blueprint** for what we build end-to-end: from data and models through interventions, uncertainty, and narrative. Diagrams use [Mermaid](https://mermaid.js.org/), which GitHub renders in markdown.

---

## 1. System overview (high level)

```mermaid
flowchart TB
  subgraph Inputs["Inputs"]
    H[Historical CRM + product usage]
    E[Optional: experiments / A-B logs]
    U[User scenario: cohort + intervention + budget]
  end

  subgraph Engines["Four engines"]
    B[Behavioral engine\nchurn / renewal + revenue @ risk]
    I[Intervention engine\nactions + causal lift assumptions]
    M[Monte Carlo simulator\nuncertainty over inputs + outcomes]
    L[LLM strategic narrator\nmemo from computed facts only]
  end

  subgraph Output["Outputs"]
    D[Distributions: saved revenue, cost, net]
    R[Recommendation + risk tradeoffs]
  end

  H --> B
  E --> I
  B --> M
  I --> M
  U --> I
  M --> D
  D --> L
  M --> L
  L --> R
```

---

## 2. Data & model path (Behavioral engine)

```mermaid
flowchart LR
  subgraph Raw["Raw data"]
    A[Accounts / subscriptions]
    C[Usage + engagement]
    S[Support + CS notes]
  end

  subgraph Features["Features"]
    F[Tabular features]
    T[Optional: text-derived features\nvia structured LLM labels]
  end

  subgraph Models["Models"]
    P[Propensity: churn or non-renewal]
    V[Value: ARPA / margin / months at risk]
  end

  A --> F
  C --> F
  S --> F
  S -.->|"optional"| T
  T --> F
  F --> P
  F --> V
```

---

## 3. Intervention & causal lift (transparent tiers)

We do **not** hide assumptions. Lift enters the simulator in explicit tiers:

```mermaid
flowchart TD
  subgraph TierA["Tier A — Demo / sensitivity"]
    A1[User-defined lift ranges\nor sliders]
    A2[Broad stress tests]
  end

  subgraph TierB["Tier B — Policy / randomized evidence"]
    B1[A-B or rollout logs]
    B2[Estimated uplift on held-out data]
  end

  subgraph TierC["Tier C — Learned uplift"]
    C1[Uplift / causal models — only with valid evaluation\nAUUC-Qini-calibration]
  end

  TierA --> SIM[Simulator inputs]
  TierB --> SIM
  TierC --> SIM
```

---

## 4. Monte Carlo loop (uncertainty → distribution)

```mermaid
flowchart TB
  START([Scenario selected]) --> PARAMS[Sample uncertain inputs:\nlift bounds, uptake, margin, baseline churn ...]
  PARAMS --> COHORT[Apply to cohort]
  COHORT --> PNL[Compute PnL: revenue at risk,\nintervention cost, net]
  PNL --> REPEAT{N runs complete?}
  REPEAT -->|no| PARAMS
  REPEAT -->|yes| DIST[Empirical distribution:\nmedian, credible intervals,\nrisk of negative net]
  DIST --> DONE([Charts + memo inputs])
```

---

## 5. LLM narrator (guardrailed “so what?”)

```mermaid
flowchart LR
  Q[Computed quantiles +\ncomparison table from code]
  Q --> SCHEMA[Structured LLM prompt\nJSON: recommendation,\nbullets, risks, assumptions]
  SCHEMA --> MEMO[Strategic memo in UI]
  Q --> VERIFY[Rendered next to charts\nnumbers must match backend]
```

**Rule:** The LLM summarizes and recommends from **frozen metrics** computed in Python—it does not invent percentiles or savings.

---

## 6. Build phases (recommended order)

| Phase | Focus |
| ------| ----- |
| **P0** | Synthetic or open dataset → Behavioral baseline → Streamlit cohort selector |
| **P1** | Intervention definitions + Tier A sensitivity + Monte Carlo + PnL rollup |
| **P2** | Optional Tier B evaluation slice + “validation” storyline |
| **P3** | Optional Tier C uplift + LLM narrator (Secrets on host, never in repo) |

---

## 7. Repository & deployment alignment (conceptual)

```mermaid
flowchart LR
  subgraph Repo["GitHub repo"]
    Code[Python + Streamlit app]
    Tests[Tests / CI optional]
  end

  subgraph CI["GitHub Actions"]
    GHS[GitHub Secrets\nfor CI only if needed]
  end

  subgraph Host["Streamlit host"]
    RunSecrets[Hosting secrets UI\nruntime OpenAI key]
  end

  Code --> Host
  GHS -.->|"pipeline only"| CI
  RunSecrets --> Host
```

Use **GitHub Secrets** for automation (tests, deploy jobs). Use the **host’s secret manager** for the live app’s OpenAI key.
