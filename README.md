# Retention Decision Lab — Next Best Action simulator

Executive-friendly **decision intelligence** demo: score renewal risk, pick a high-value cohort, compare two retention plays, and read off **distributions of net margin** (not a single lucky number). Optional GPT memo restates the **Python-computed** quantiles only.

---

## Why this exists (portfolio story)

- **Behavioral engine:** interpretable churn risk on a documented synthetic CRM generator (honest about it being synthetic).  
- **Intervention engine (Tier A):** lift and uptake are **sensitivity ranges**—the UI makes that explicit.  
- **Monte Carlo:** uncertainty over those inputs becomes **curves and tail risk** (e.g., probability the play loses money).  
- **Narrative layer:** OpenAI summarizes tradeoffs against the JSON emitted by code—numbers are never free-hand by the LLM.

See also [`PROJECT_FLOW.md`](PROJECT_FLOW.md) for diagrams and [`MODEL_AND_SOLUTION.md`](MODEL_AND_SOLUTION.md) for the business framing.

---

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Copy secrets template → .streamlit/secrets.toml and add OPENAI_API_KEY if you want the memo tab
streamlit run app.py
```

Default port: Streamlit prints a `localhost` URL.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (no secrets in the tree).  
2. New app → select repo → main file `app.py`.  
3. **Secrets** in the Cloud UI: add `OPENAI_API_KEY` for the memo tab.  
4. `requirements.txt` is picked up automatically.

**About GitHub Secrets:** they power **Actions/CI**, not the running Streamlit process. For production runtime keys, use Streamlit’s secret manager (or your container platform’s equivalent).

---

## Repo map

| File | Role |
| --- | --- |
| `app.py` | Streamlit UX, charts, scenario controls |
| `model.py` | Synthetic data DGP, logistic baseline, Monte Carlo engine |
| `narrative.py` | Optional JSON-only GPT memo |
| `PROJECT_FLOW.md` | Architecture / flow diagrams |
| `MODEL_AND_SOLUTION.md` | Product + modeling narrative |
| `.streamlit/config.toml` | Theme + light-weight defaults |
| `.streamlit/secrets.toml.example` | Template for local secrets |

---

## Interview talking points (60 seconds)

1. **Problem:** leaders need *decision* metrics, not only propensity scores.  
2. **Approach:** combine calibrated-enough risk with explicit economic rollups and bandit-friendly scenario math.  
3. **Honesty:** synthetic data + Tier A lift → position as *sensitivity lab*; next step is experiments / uplift evaluation.  
4. **AI:** LLM is a **presenter** over structured metrics, not the source of truth for dollars.

---

## License

Use freely for portfolio and interviews; attribute if you fork publicly.
