from __future__ import annotations

import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model as mdl
from narrative import synthesize_memo


st.set_page_config(
    page_title="Retention Decision Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Calibrating synthetic workspace…")
def load_workspace(database_seed: int, horizon_months: int, n_accounts: int) -> tuple[pd.DataFrame, mdl.ModelBundle]:
    rng_train = database_seed + 11
    raw = mdl.generate_synthetic_crm_data(n_customers=n_accounts, seed=database_seed, renewal_horizon_months=horizon_months)
    bundle = mdl.train_churn_model(raw, seed=rng_train % 5000)
    scored = mdl.score_accounts(raw, bundle)
    return scored, bundle


def intervention_presets(which: str) -> mdl.InterventionSpec:
    if which == "Renewal concession (priced in)":
        return mdl.InterventionSpec(
            name="Renewal concession",
            uplift_pp_low=0.022,
            uplift_pp_high=0.052,
            uptake_low=0.48,
            uptake_high=0.72,
            fixed_cost_per_touched_account=12.5,
            discount_rate_on_mrr=0.12,
            discount_horizon_months=12,
        )
    if which == "High-touch save play":
        return mdl.InterventionSpec(
            name="High-touch save play",
            uplift_pp_low=0.032,
            uplift_pp_high=0.068,
            uptake_low=0.35,
            uptake_high=0.58,
            fixed_cost_per_touched_account=190.0,
            discount_rate_on_mrr=0.0,
            discount_horizon_months=0,
        )
    if which == "Enablement burst (cheap, weak lift)":
        return mdl.InterventionSpec(
            name="Enablement burst",
            uplift_pp_low=0.015,
            uplift_pp_high=0.029,
            uptake_low=0.55,
            uptake_high=0.78,
            fixed_cost_per_touched_account=40.0,
            discount_rate_on_mrr=0.0,
            discount_horizon_months=0,
        )
    return mdl.InterventionSpec(name=str(which))


def distribution_figure(results_a: dict, results_b: dict, label_a: str, label_b: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=results_a["net_benefit"],
            name=label_a,
            opacity=0.55,
            nbinsx=36,
            histnorm="probability density",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=results_b["net_benefit"],
            name=label_b,
            opacity=0.55,
            nbinsx=36,
            histnorm="probability density",
        )
    )
    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Net margin impact vs do-nothing (USD)",
        yaxis_title="Density",
    )
    return fig


def calibration_figure(bundle: mdl.ModelBundle) -> go.Figure | None:
    cal = bundle.metrics.get("calibration") if bundle.metrics else None
    if not cal:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cal["prob_pred"],
            y=cal["prob_true"],
            mode="lines+markers",
            name="Reliability",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash", color="#b0b0b0"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        xaxis_title="Mean predicted risk (bin)",
        yaxis_title="Observed churn rate",
        margin=dict(l=20, r=20, t=30, b=30),
    )
    return fig


def coefficient_figure(bundle: mdl.ModelBundle) -> go.Figure | None:
    coefs = bundle.metrics.get("logistic_coef_scaled") if bundle.metrics else None
    if not coefs:
        return None
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#4c78a8"))
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title="Coefficient (standardized features)",
        yaxis_title="",
    )
    return fig


def _openai_api_key() -> str:
    try:
        val = st.secrets["OPENAI_API_KEY"]
    except Exception:  # noqa: BLE001
        return ""
    return val if isinstance(val, str) else str(val)


def main() -> None:
    st.title("Retention Decision Lab")
    st.caption("Simulate **net margin** outcomes under uncertain interventions — not only churn probability.")

    with st.sidebar:
        st.header("Inputs")
        database_seed = st.number_input("Synthetic data seed", min_value=0, max_value=99_999, value=42, step=1)
        n_accounts = st.slider("Accounts in universe", min_value=1500, max_value=8000, value=4000, step=250)
        horizon_months = st.slider("Renewal horizon (months)", min_value=2, max_value=6, value=3)
        sim_runs = st.slider("Monte Carlo draws", min_value=150, max_value=900, value=360, step=30)
        st.divider()
        st.markdown("**Cohort selection**")
        top_n = st.slider("Priority accounts to touch", min_value=150, max_value=1200, value=450, step=50)
        segment_filter = st.multiselect(
            "Segment filter (optional)",
            options=["SMB", "Mid-market", "Strategic"],
            default=["SMB", "Mid-market", "Strategic"],
        )
        st.divider()
        st.markdown(
            "Tier A lift is **sensitivity**, not a causal guarantee. "
            "Tighten ranges with experiments when you have them."
        )

    scored, bundle = load_workspace(int(database_seed), int(horizon_months), int(n_accounts))

    segments = tuple(segment_filter) if segment_filter else None
    cohort = mdl.prioritize_cohort(scored, top_n=top_n, segment_filter=segments)
    if cohort.empty:
        cohort = scored.nlargest(top_n, "expected_margin_loss_horizon").copy()

    baseline_loss = float((cohort["p_churn_horizon"] * cohort["margin_at_risk_horizon"]).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts scored", f"{len(scored):,}")
    roc = bundle.metrics["roc_auc"]
    roc_label = f"{roc:.3f}" if isinstance(roc, (int, float)) and math.isfinite(roc) else "n/a"
    k2.metric("Holdout ROC-AUC", roc_label)
    k3.metric("Empirical churn rate", f"{scored['churn_within_horizon'].mean():.1%}")
    k4.metric("Selected cohort expected loss", mdl.format_currency(baseline_loss))

    tab_overview, tab_sim, tab_model, tab_memo = st.tabs(["Overview", "Decision simulator", "Model card", "Strategic memo"])

    with tab_overview:
        st.markdown(
            """
**What this demonstrates:** a behavioral risk model, transparent intervention assumptions, and Monte Carlo uncertainty.
You compare two plays on the *same at-risk cohort* and read off upside *and* tail risk.
            """.strip()
        )
        st.divider()
        left, right = st.columns((1.1, 0.9))
        with left:
            st.subheader("Top of queue (sample)")
            display_cols = [
                "segment",
                "p_churn_horizon",
                "margin_at_risk_horizon",
                "expected_margin_loss_horizon",
                "login_events_30d",
                "support_tickets_90d",
                "nps_score",
            ]
            sample = cohort[display_cols].head(12).copy()
            sample["p_churn_horizon_fmt"] = sample["p_churn_horizon"].map(lambda v: f"{v:.1%}")
            sample["margin_at_risk_fmt"] = sample["margin_at_risk_horizon"].map(mdl.format_currency)
            sample["expected_loss_fmt"] = sample["expected_margin_loss_horizon"].map(mdl.format_currency)
            show_cols = [
                "segment",
                "p_churn_horizon_fmt",
                "margin_at_risk_fmt",
                "expected_loss_fmt",
                "login_events_30d",
                "support_tickets_90d",
                "nps_score",
            ]
            st.dataframe(sample[show_cols].rename(columns={"p_churn_horizon_fmt": "p_churn", "margin_at_risk_fmt": "margin@risk", "expected_loss_fmt": "exp_loss"}), use_container_width=True, height=360)
        with right:
            st.subheader("Expected loss footprint")
            fig_loss = go.Figure(
                go.Histogram(x=scored["expected_margin_loss_horizon"], nbinsx=45, marker_color="#72b7b2")
            )
            fig_loss.update_layout(
                template="plotly_white", height=360, xaxis_title="Expected margin loss", yaxis_title="Accounts"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    with tab_sim:
        st.subheader("Configure competing interventions")
        c1, c2 = st.columns(2)
        preset_a = c1.selectbox(
            "Scenario A preset",
            [
                "Renewal concession (priced in)",
                "High-touch save play",
                "Enablement burst (cheap, weak lift)",
            ],
            index=0,
        )
        preset_b = c2.selectbox(
            "Scenario B preset",
            ["High-touch save play", "Renewal concession (priced in)", "Enablement burst (cheap, weak lift)"],
            index=0,
        )

        spec_a = intervention_presets(preset_a)
        spec_b = intervention_presets(preset_b)

        with st.expander("Fine-tune Scenario A", expanded=False):
            sa1, sa2, sa3 = st.columns(3)
            spec_a.uplift_pp_low = float(sa1.number_input("Lift low (pp)", value=float(spec_a.uplift_pp_low), step=0.005, format="%.3f"))
            spec_a.uplift_pp_high = float(sa2.number_input("Lift high (pp)", value=float(spec_a.uplift_pp_high), step=0.005, format="%.3f"))
            spec_a.uptake_low = float(sa3.number_input("Uptake low", value=float(spec_a.uptake_low), step=0.02, format="%.2f"))
            spec_a.uptake_high = float(sa1.number_input("Uptake high", value=float(spec_a.uptake_high), step=0.02, format="%.2f"))
            spec_a.fixed_cost_per_touched_account = float(sa2.number_input("Fixed $ per touched account", value=float(spec_a.fixed_cost_per_touched_account), step=5.0))
            spec_a.discount_rate_on_mrr = float(sa3.number_input("Discount rate on MRR", value=float(spec_a.discount_rate_on_mrr), step=0.01, format="%.2f"))
            spec_a.discount_horizon_months = int(sa1.number_input("Discount horizon (months)", value=int(spec_a.discount_horizon_months), min_value=0, max_value=36, step=1))

        with st.expander("Fine-tune Scenario B", expanded=False):
            sb1, sb2, sb3 = st.columns(3)
            spec_b.uplift_pp_low = float(sb1.number_input("B lift low (pp)", value=float(spec_b.uplift_pp_low), step=0.005, format="%.3f"))
            spec_b.uplift_pp_high = float(sb2.number_input("B lift high (pp)", value=float(spec_b.uplift_pp_high), step=0.005, format="%.3f"))
            spec_b.uptake_low = float(sb3.number_input("B uptake low", value=float(spec_b.uptake_low), step=0.02, format="%.2f"))
            spec_b.uptake_high = float(sb1.number_input("B uptake high", value=float(spec_b.uptake_high), step=0.02, format="%.2f"))
            spec_b.fixed_cost_per_touched_account = float(sb2.number_input("B fixed $ per touched account", value=float(spec_b.fixed_cost_per_touched_account), step=5.0))
            spec_b.discount_rate_on_mrr = float(sb3.number_input("B discount rate on MRR", value=float(spec_b.discount_rate_on_mrr), step=0.01, format="%.2f"))
            spec_b.discount_horizon_months = int(sb1.number_input("B discount horizon (months)", value=int(spec_b.discount_horizon_months), min_value=0, max_value=36, step=1))

        run = st.button("Run Monte Carlo comparison", type="primary", use_container_width=True)
        if run:
            with st.spinner("Running draws…"):
                res_a = mdl.simulate_intervention_distribution(cohort, spec_a, n_simulations=sim_runs, seed=int(database_seed) + 3)
                res_b = mdl.simulate_intervention_distribution(cohort, spec_b, n_simulations=sim_runs, seed=int(database_seed) + 5)
                st.session_state["last_sim_a"] = res_a
                st.session_state["last_sim_b"] = res_b
                st.session_state["last_spec_names"] = (spec_a.name, spec_b.name)

        res_a_state = st.session_state.get("last_sim_a")
        res_b_state = st.session_state.get("last_sim_b")
        if res_a_state and res_b_state:
            names = st.session_state.get("last_spec_names", (spec_a.name, spec_b.name))
            sum_a = mdl.summarize_draws(res_a_state["net_benefit"], names[0])
            sum_b = mdl.summarize_draws(res_b_state["net_benefit"], names[1])

            qa, qb = st.columns(2)
            qa.markdown(f"##### {names[0]} — net vs do nothing")
            qb.markdown(f"##### {names[1]} — net vs do nothing")
            m1a, m2a, m3a = qa.columns(3)
            m1b, m2b, m3b = qb.columns(3)
            m1a.metric("Median", mdl.format_currency(sum_a["median"]))
            m2a.metric("5th–95th", f"{mdl.format_currency(sum_a['p05'])}→{mdl.format_currency(sum_a['p95'])}")
            m3a.metric("P(net loss)", f"{sum_a['prob_negative']:.0%}")

            m1b.metric("Median", mdl.format_currency(sum_b["median"]))
            m2b.metric("5th–95th", f"{mdl.format_currency(sum_b['p05'])}→{mdl.format_currency(sum_b['p95'])}")
            m3b.metric("P(net loss)", f"{sum_b['prob_negative']:.0%}")

            st.plotly_chart(distribution_figure(res_a_state, res_b_state, names[0], names[1]), use_container_width=True)
            st.caption("Tail risk frequently comes from discount economics or tepid uptake — not only scoring noise.")

            st.session_state["memo_context"] = {
                "cohort_accounts": int(cohort.shape[0]),
                "baseline_expected_margin_loss": round(baseline_loss, 2),
                "monte_carlo_runs": int(sim_runs),
                "scenario_a": {
                    "name": names[0],
                    "summary": {key: round(value, 4) for key, value in sum_a.items() if key != "label"},
                },
                "scenario_b": {
                    "name": names[1],
                    "summary": {key: round(value, 4) for key, value in sum_b.items() if key != "label"},
                },
            }
        else:
            st.info("Run the Monte Carlo sweep to populate charts plus memo inputs.")

    with tab_model:
        st.subheader("Transparent churn baseline")
        roc = bundle.metrics["roc_auc"]
        roc_label = f"{roc:.3f}" if isinstance(roc, (int, float)) and math.isfinite(roc) else "n/a"
        st.write(f"Training size {bundle.metrics['n_train']:,} · Held-out ROC-AUC {roc_label}")

        cmid, cright = st.columns(2)
        cal_fig = calibration_figure(bundle)
        coef_fig = coefficient_figure(bundle)
        if cal_fig:
            cmid.plotly_chart(cal_fig, use_container_width=True)
        if coef_fig:
            cright.plotly_chart(coef_fig, use_container_width=True)

        coef_df = pd.DataFrame([{"feature": k, "coefficient_scaled": v} for k, v in bundle.metrics["logistic_coef_scaled"].items()])
        with st.expander("Coefficient table"):
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
**Sandbox caveats**

- Synthetic generator ⇒ metrics sanity-check fidelity, not market truth  
- Associations only unless you bolt on uplift evaluation with experiments  
- Margin roll-ups compress rich contract nuances into horizon dollars
            """.strip()
        )

    with tab_memo:
        st.markdown("Optional GPT narration uses **frozen JSON facts** emitted by Python — no invented quantiles.")

        api_key = _openai_api_key().strip()
        if not api_key:
            st.warning("Add `OPENAI_API_KEY` through Streamlit secrets or `.streamlit/secrets.toml` (never commit it).")

        memo_ctx = st.session_state.get("memo_context")
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
        memo_btn = st.button("Generate memo", disabled=not memo_ctx or not api_key)

        if memo_btn and memo_ctx and api_key:
            with st.spinner("Drafting structured memo…"):
                try:
                    memo_obj = synthesize_memo(metrics_context=memo_ctx, api_key=api_key, model=model_name.strip())
                    st.subheader(memo_obj.get("headline", "Decision memo"))
                    for bullet in memo_obj.get("bullets", []):
                        st.markdown(f"- {bullet}")
                    st.markdown("**Recommendation:** " + str(memo_obj.get("recommend", "neutral")).replace("_", " "))
                    risks = memo_obj.get("risks", [])
                    if risks:
                        st.markdown("**Risks or caveats**")
                        for item in risks:
                            st.markdown(f"- {item}")
                    if memo_obj.get("confidence_note"):
                        st.caption(memo_obj["confidence_note"])
                    with st.expander("Authoritative metrics JSON"):
                        st.code(json.dumps(memo_ctx, indent=2))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Memo generation failed: {exc}")

        elif not memo_ctx:
            st.info("Run the simulator tab first so quantiles exist to narrate.")


if __name__ == "__main__":
    main()
