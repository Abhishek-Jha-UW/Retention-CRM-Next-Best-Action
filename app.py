from __future__ import annotations

import io
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import streamlit as st

import churn_pipeline as cp
import model as mdl
from narrative import explain_model_performance, synthesize_memo


st.set_page_config(
    page_title="Retention Decision Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session() -> None:
    defaults = {
        "raw_df": None,
        "trained": None,
        "scored_df": None,
        "sim_df": None,
        "churn_trained": False,
        "model_explain_cache": None,
        "last_sim_a": None,
        "last_sim_b": None,
        "last_spec_names": None,
        "memo_context": None,
        "cm_threshold": 0.5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _openai_api_key() -> str:
    try:
        val = st.secrets["OPENAI_API_KEY"]
    except Exception:  # noqa: BLE001
        return ""
    return val if isinstance(val, str) else str(val)


def intervention_presets(which: str) -> mdl.InterventionSpec:
    if which == "Renewal concession (priced in)":
        return mdl.InterventionSpec(
            name="Renewal concession",
            uplift_pp_low=0.022,
            uplift_pp_high=0.052,
            uptake_low=0.42,
            uptake_high=0.62,
            fixed_cost_per_touched_account=12.5,
            discount_rate_on_mrr=0.08,
            discount_horizon_months=6,
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
    fig.add_vline(x=0.0, line_dash="dash", line_color="#64748b", annotation_text="Do nothing", annotation_position="top")
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


def confusion_figure(cm: dict) -> go.Figure:
    # Rows: actual 0, 1 — cols: predicted 0, 1
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    text = [[f"TN<br>{cm['tn']}", f"FP<br>{cm['fp']}"], [f"FN<br>{cm['fn']}", f"TP<br>{cm['tp']}"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"Confusion @ threshold {cm.get('threshold', 0.5):.2f}",
    )
    return fig


def calibration_holdout_figure(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=8, strategy="quantile")
    except ValueError:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Reliability"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="#94a3b8")))
    fig.update_layout(
        template="plotly_white",
        height=320,
        title="Holdout calibration (quantile bins)",
        xaxis_title="Mean predicted risk",
        yaxis_title="Observed churn rate",
        margin=dict(l=20, r=20, t=40, b=40),
    )
    return fig


def roc_operating_point(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom_neg = fp + tn
    denom_pos = tp + fn
    fpr = float(fp / denom_neg) if denom_neg else 0.0
    tpr = float(tp / denom_pos) if denom_pos else 0.0
    return fpr, tpr


def roc_figure(
    curve: dict,
    auc: float,
    threshold: float | None = None,
    y_true: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    if curve.get("fpr"):
        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name="ROC"))
    if threshold is not None and y_true is not None and y_score is not None and len(np.unique(y_true)) >= 2:
        fpr_op, tpr_op = roc_operating_point(y_true, y_score, threshold)
        fig.add_trace(
            go.Scatter(
                x=[fpr_op],
                y=[tpr_op],
                mode="markers",
                marker=dict(size=12, color="#dc2626", symbol="circle"),
                name=f"Holdout @ τ={threshold:.2f}",
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#999")))
    auc_txt = f"{auc:.3f}" if isinstance(auc, (int, float)) and math.isfinite(auc) else "n/a"
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=20, r=20, t=40, b=40),
        title=f"ROC curve (holdout AUC = {auc_txt})",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    return fig


def coef_or_importance_figure(trained: cp.TrainedChurnModel) -> go.Figure | None:
    m = trained.metrics
    if trained.model_kind == "logistic" and m.get("coefficients"):
        items = sorted(m["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        title = "Logistic coefficients (transformed feature space)"
    elif trained.model_kind == "xgboost" and m.get("feature_importance"):
        items = sorted(m["feature_importance"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        title = "XGBoost feature importance (gain-based)"
    else:
        return None
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#4c78a8"))
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=40, b=30), title=title)
    return fig


def _is_builtin_sample_schema(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in mdl.FEATURE_COLUMNS) and "churn_within_horizon" in df.columns


def main() -> None:
    _init_session()
    st.title("Retention Decision Lab")
    st.caption("Train churn on **your** rows, then simulate how interventions move **net margin** under uncertainty.")

    api_key = _openai_api_key().strip()

    with st.sidebar:
        st.header("Session")
        if st.button("Reset workspace", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_session()
            st.rerun()
        st.divider()
        st.markdown("**Scenario Monte Carlo**")
        sim_runs = st.slider("Monte Carlo draws", min_value=150, max_value=900, value=360, step=30)
        top_n = st.slider("Priority accounts to touch", min_value=50, max_value=1200, value=450, step=25)
        st.divider()
        st.caption("Tier A lift = sensitivity bands, not proven causal lift without experiments.")

    st.markdown("### Step 1 — Load data and train a churn model")
    st.markdown(
        "Upload a labeled table (historical churn / renewal outcome as 0/1), pick features, "
        "then review discrimination (ROC-AUC), calibration-at-threshold (confusion matrix), and drivers."
    )

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button(
            label="Download CSV template",
            data=cp.build_template_csv(),
            file_name="churn_training_template.csv",
            mime="text/csv",
        )
    with dc2:
        if st.button("Load sample CRM data", type="primary"):
            st.session_state["raw_df"] = mdl.generate_synthetic_crm_data(n_customers=3500, seed=42, renewal_horizon_months=3)
            st.session_state["churn_trained"] = False
            st.session_state["trained"] = None
            st.session_state["scored_df"] = None
            st.session_state["sim_df"] = None
            st.success("Sample loaded — scroll to map columns and train.")
            st.rerun()

    with dc3:
        up = st.file_uploader("Upload CSV", type=["csv"], help="UTF-8 CSV with a binary churn column.")
        if up is not None:
            try:
                st.session_state["raw_df"] = pd.read_csv(up)
                st.session_state["churn_trained"] = False
                st.session_state["trained"] = None
                st.session_state["scored_df"] = None
                st.session_state["sim_df"] = None
                st.success(f"Loaded {len(st.session_state['raw_df']):,} rows.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse CSV: {exc}")

    paste = st.text_area("Or paste CSV here (include header row)", height=140, placeholder="account_id,churn,...")
    if st.button("Parse pasted CSV"):
        if not paste.strip():
            st.warning("Paste CSV text first.")
        else:
            try:
                st.session_state["raw_df"] = pd.read_csv(io.StringIO(paste))
                st.session_state["churn_trained"] = False
                st.session_state["trained"] = None
                st.session_state["scored_df"] = None
                st.session_state["sim_df"] = None
                st.success(f"Parsed {len(st.session_state['raw_df']):,} rows.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Parse error: {exc}")

    raw_df: pd.DataFrame | None = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Load sample data, upload a file, or paste CSV to continue.")
        return

    st.caption(f"Current table: **{len(raw_df):,}** rows × **{raw_df.shape[1]}** columns.")

    with st.expander("Preview data", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Column mapping")
    cols_list = raw_df.columns.tolist()
    id_exclude = st.multiselect("ID / leakage columns to exclude from features", options=cols_list, default=[c for c in cols_list if c.lower() in {"account_id", "id", "customer_id"}])
    target_col = st.selectbox("Churn label column (0 = retained, 1 = churned)", options=cols_list, index=cols_list.index("churn_within_horizon") if "churn_within_horizon" in cols_list else 0)

    suggested = cp.suggest_numeric_feature_columns(raw_df, target_col, id_exclude)
    if _is_builtin_sample_schema(raw_df):
        default_feats = [c for c in mdl.FEATURE_COLUMNS if c in suggested]
    else:
        default_feats = suggested[: min(12, len(suggested))]
    if not default_feats:
        default_feats = suggested[: min(8, len(suggested))]
    feature_cols = st.multiselect("Feature columns (numeric)", options=suggested, default=default_feats or suggested[:8])
    if len(feature_cols) < 2:
        st.warning("Pick at least two numeric feature columns.")

    model_choice_ui = st.selectbox(
        "Model",
        options=["Automatic", "Logistic regression", "XGBoost"],
        index=0,
        help="Automatic compares holdout ROC-AUC between logistic and XGBoost when xgboost is installed.",
    )
    test_size = st.slider("Holdout fraction", min_value=0.15, max_value=0.4, value=0.25, step=0.05)
    train_seed = st.number_input("Train / split seed", min_value=0, max_value=99_999, value=42, step=1)

    train_btn = st.button("Train churn model", type="primary", disabled=len(feature_cols) < 2)
    if train_btn:
        try:
            choice_map = {"Automatic": "automatic", "Logistic regression": "logistic", "XGBoost": "xgboost"}
            trained = cp.train_churn_classifier(
                raw_df,
                feature_columns=feature_cols,
                target_column=target_col,
                model_choice=choice_map[model_choice_ui],
                test_size=float(test_size),
                seed=int(train_seed),
            )
            scored = cp.score_full_dataframe(raw_df, trained)
            st.session_state["trained"] = trained
            st.session_state["scored_df"] = scored
            st.session_state["churn_trained"] = True
            st.session_state["sim_df"] = None
            st.session_state["model_explain_cache"] = None
            st.session_state["last_sim_a"] = None
            st.session_state["last_sim_b"] = None
            st.session_state["memo_context"] = None
            st.session_state["cm_threshold"] = 0.5
            st.success("Model trained — review metrics below, then continue to scenarios.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")

    trained = st.session_state.get("trained")
    if not st.session_state.get("churn_trained") or trained is None:
        return

    st.divider()
    m = trained.metrics
    auc = m["roc_auc_holdout"]
    auc_label = f"{auc:.3f}" if isinstance(auc, (int, float)) and math.isfinite(auc) else "n/a"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", trained.model_kind.title())
    m2.metric("Holdout ROC-AUC", auc_label)
    m3.metric("Train rows", f"{m['n_train']:,}")
    m4.metric("Test churn rate", f"{m['test_churn_rate']:.1%}")

    if note := m.get("auto_selection_note"):
        st.info(note)

    st.markdown("#### Operating threshold (confusion matrix)")
    st.caption(
        "ROC-AUC summarizes all cutoffs; the confusion matrix is **one** cutoff. "
        "Move the slider to explore precision/recall tradeoffs (e.g. fewer false alarms vs. missed churners)."
    )
    st.slider(
        "Classify as churn if P(churn) ≥",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("cm_threshold", 0.5)),
        step=0.01,
        key="cm_threshold",
    )
    thr = float(st.session_state["cm_threshold"])

    y_h = getattr(trained, "y_holdout", None)
    p_h = getattr(trained, "proba_holdout", None)
    if y_h is not None and p_h is not None:
        cm_dyn = cp.confusion_matrix_at_threshold(y_h, p_h, thr)
        tmet = cp.threshold_classification_metrics(y_h, p_h, thr)
        kpr, krc, kf1, kfg = st.columns(4)
        kpr.metric("Precision (churn=1)", f"{tmet['precision']:.3f}")
        krc.metric("Recall (churn=1)", f"{tmet['recall']:.3f}")
        kf1.metric("F1 (churn=1)", f"{tmet['f1']:.3f}")
        kfg.metric("Flagged as high-risk", f"{tmet['flagged_rate']:.1%}")
    else:
        cm_dyn = m["confusion_matrix"]

    cleft, cright = st.columns(2)
    with cleft:
        st.plotly_chart(confusion_figure(cm_dyn), use_container_width=True)
    with cright:
        st.plotly_chart(
            roc_figure(
                m["roc_curve"],
                m["roc_auc_holdout"],
                threshold=thr,
                y_true=y_h,
                y_score=p_h,
            ),
            use_container_width=True,
        )

    if y_h is not None and p_h is not None:
        cal_fig = calibration_holdout_figure(y_h, p_h)
        if cal_fig:
            st.plotly_chart(cal_fig, use_container_width=True)

    fig_drv = coef_or_importance_figure(trained)
    if fig_drv:
        st.plotly_chart(fig_drv, use_container_width=True)

    st.markdown("#### AI: interpret model performance")
    if not api_key:
        st.warning("Add `OPENAI_API_KEY` in Streamlit secrets to enable explanations.")
    else:
        _yh = getattr(trained, "y_holdout", None)
        _ph = getattr(trained, "proba_holdout", None)
        cm_ai = (
            cp.confusion_matrix_at_threshold(_yh, _ph, float(st.session_state["cm_threshold"]))
            if _yh is not None and _ph is not None
            else m.get("confusion_matrix")
        )
        tmet_ai = (
            cp.threshold_classification_metrics(_yh, _ph, float(st.session_state["cm_threshold"]))
            if _yh is not None and _ph is not None
            else {}
        )
        explain_ctx = {
            "model_kind": trained.model_kind,
            "feature_columns": list(trained.feature_columns),
            "target_column": trained.target_column,
            "roc_auc_holdout": m.get("roc_auc_holdout"),
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
            "test_churn_rate": m.get("test_churn_rate"),
            "threshold_used": float(st.session_state["cm_threshold"]),
            "confusion_matrix": cm_ai,
            "threshold_metrics": tmet_ai,
        }
        if st.button("Explain metrics with AI", key="explain_model_ai"):
            with st.spinner("Summarizing…"):
                try:
                    st.session_state["model_explain_cache"] = explain_model_performance(explain_ctx, api_key=api_key)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        cache = st.session_state.get("model_explain_cache")
        if cache:
            st.markdown(f"**{cache.get('summary', '')}**")
            if cache.get("strengths"):
                st.markdown("**Strengths**")
                for b in cache["strengths"]:
                    st.markdown(f"- {b}")
            if cache.get("watchouts"):
                st.markdown("**Watchouts**")
                for b in cache["watchouts"]:
                    st.markdown(f"- {b}")
            if cache.get("next_steps"):
                st.markdown("**Next steps**")
                for b in cache["next_steps"]:
                    st.markdown(f"- {b}")

    st.markdown("---")
    st.markdown("### Step 2 — Map economics, then run revenue & scenario simulations")
    st.success(
        "Churn scores are ready. **Next:** tie each row to **dollar exposure** (margin or revenue at risk) "
        "so we can rank accounts and stress-test interventions."
    )

    scored_df: pd.DataFrame = st.session_state["scored_df"]
    exclude_econ = {"p_churn_horizon", target_col}
    num_choices = [
        c
        for c in scored_df.columns
        if pd.api.types.is_numeric_dtype(scored_df[c]) and c not in exclude_econ
    ]
    if not num_choices:
        st.error("No numeric columns left for economics mapping — check your data.")
        return

    default_margin = "margin_at_risk_horizon" if "margin_at_risk_horizon" in scored_df.columns else num_choices[0]
    default_mrr = "mrr_monthly" if "mrr_monthly" in scored_df.columns else None
    default_seg = "segment" if "segment" in scored_df.columns else None

    em1, em2, em3 = st.columns(3)
    margin_col = em1.selectbox(
        "Value column ($ at risk if churn)",
        options=num_choices,
        index=num_choices.index(default_margin) if default_margin in num_choices else 0,
        help="Either total margin dollars already rolled to the horizon, or pick monthly and check the box below.",
    )
    margin_is_monthly = em2.checkbox("Column is monthly margin (multiply by horizon)", value=False)
    horizon_econ = int(em3.number_input("Horizon (months) for monthly → total", min_value=1, max_value=36, value=3, step=1))
    mrr_opts = ["(derive from margin / horizon)"] + [c for c in num_choices if c != margin_col]
    default_mrr_idx = 0
    if default_mrr and default_mrr in mrr_opts:
        default_mrr_idx = mrr_opts.index(default_mrr)
    mrr_col = st.selectbox(
        "MRR column (for discount costing; optional)",
        options=mrr_opts,
        index=default_mrr_idx,
    )
    seg_col = st.selectbox("Segment column (optional)", options=["(none — single segment)"] + cols_list, index=0)
    mrr_pick = None if mrr_col.startswith("(") else mrr_col
    seg_pick = None if seg_col.startswith("(") else seg_col

    if st.button("Apply economics & unlock scenarios", type="primary"):
        try:
            sim_df = mdl.attach_simulation_economics(
                scored_df,
                margin_column=margin_col,
                margin_is_monthly=margin_is_monthly,
                horizon_months=horizon_econ,
                mrr_column=mrr_pick,
                segment_column=seg_pick,
            )
            st.session_state["sim_df"] = sim_df
            st.session_state["horizon_econ"] = horizon_econ
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    sim_df: pd.DataFrame | None = st.session_state.get("sim_df")
    if sim_df is None:
        st.info("Click **Apply economics & unlock scenarios** to continue.")
        return

    st.markdown(
        "#### What happens here\n"
        "We rank accounts by **expected margin loss** = churn probability × dollars at risk, "
        "then Monte Carlo sample uncertain **lift** and **uptake** bands for two interventions. "
        "Distributions show upside *and* tail risk (e.g., costly discounts with weak uptake)."
    )

    seg_opts = sorted(sim_df["segment"].astype(str).unique().tolist())
    segment_filter = st.multiselect("Segment filter for touched cohort", options=seg_opts, default=seg_opts)

    segments = tuple(segment_filter) if segment_filter else None
    cohort = mdl.prioritize_cohort(sim_df, top_n=top_n, segment_filter=segments)
    if cohort.empty:
        cohort = sim_df.nlargest(top_n, "expected_margin_loss_horizon").copy()

    baseline_loss = float((cohort["p_churn_horizon"] * cohort["margin_at_risk_horizon"]).sum())
    s1, s2, s3 = st.columns(3)
    s1.metric("Scored accounts", f"{len(sim_df):,}")
    s2.metric("Cohort size", f"{len(cohort):,}")
    s3.metric("Cohort expected margin loss", mdl.format_currency(baseline_loss))

    st.subheader("Cohort snapshot")
    show_cols = [c for c in ["segment", "p_churn_horizon", "margin_at_risk_horizon", "expected_margin_loss_horizon"] if c in cohort.columns]
    snap = cohort[show_cols].head(15).copy()
    if "p_churn_horizon" in snap.columns:
        snap["p_churn_horizon"] = snap["p_churn_horizon"].map(lambda v: f"{v:.1%}")
    for col in ("margin_at_risk_horizon", "expected_margin_loss_horizon"):
        if col in snap.columns:
            snap[col] = snap[col].map(mdl.format_currency)
    st.dataframe(snap, use_container_width=True, hide_index=True)

    st.subheader("Scenario Monte Carlo")
    c1, c2 = st.columns(2)
    preset_a = c1.selectbox(
        "Scenario A",
        [
            "Renewal concession (priced in)",
            "High-touch save play",
            "Enablement burst (cheap, weak lift)",
        ],
        index=0,
        key="pa",
    )
    preset_b = c2.selectbox(
        "Scenario B",
        ["High-touch save play", "Renewal concession (priced in)", "Enablement burst (cheap, weak lift)"],
        index=0,
        key="pb",
    )
    spec_a = intervention_presets(preset_a)
    spec_b = intervention_presets(preset_b)

    with st.expander("Tune Scenario A assumptions", expanded=False):
        a1, a2, a3 = st.columns(3)
        spec_a.uplift_pp_low = float(a1.number_input("A lift low (pp)", value=float(spec_a.uplift_pp_low), step=0.005, format="%.3f", key="a1"))
        spec_a.uplift_pp_high = float(a2.number_input("A lift high (pp)", value=float(spec_a.uplift_pp_high), step=0.005, format="%.3f", key="a2"))
        spec_a.uptake_low = float(a3.number_input("A uptake low", value=float(spec_a.uptake_low), step=0.02, format="%.2f", key="a3"))
        spec_a.uptake_high = float(a1.number_input("A uptake high", value=float(spec_a.uptake_high), step=0.02, format="%.2f", key="a4"))
        spec_a.fixed_cost_per_touched_account = float(a2.number_input("A fixed $ / account", value=float(spec_a.fixed_cost_per_touched_account), step=5.0, key="a5"))
        spec_a.discount_rate_on_mrr = float(a3.number_input("A discount on MRR", value=float(spec_a.discount_rate_on_mrr), step=0.01, format="%.2f", key="a6"))
        spec_a.discount_horizon_months = int(a1.number_input("A discount months", value=int(spec_a.discount_horizon_months), min_value=0, max_value=36, step=1, key="a7"))

    with st.expander("Tune Scenario B assumptions", expanded=False):
        b1, b2, b3 = st.columns(3)
        spec_b.uplift_pp_low = float(b1.number_input("B lift low (pp)", value=float(spec_b.uplift_pp_low), step=0.005, format="%.3f", key="b1"))
        spec_b.uplift_pp_high = float(b2.number_input("B lift high (pp)", value=float(spec_b.uplift_pp_high), step=0.005, format="%.3f", key="b2"))
        spec_b.uptake_low = float(b3.number_input("B uptake low", value=float(spec_b.uptake_low), step=0.02, format="%.2f", key="b3"))
        spec_b.uptake_high = float(b1.number_input("B uptake high", value=float(spec_b.uptake_high), step=0.02, format="%.2f", key="b4"))
        spec_b.fixed_cost_per_touched_account = float(b2.number_input("B fixed $ / account", value=float(spec_b.fixed_cost_per_touched_account), step=5.0, key="b5"))
        spec_b.discount_rate_on_mrr = float(b3.number_input("B discount on MRR", value=float(spec_b.discount_rate_on_mrr), step=0.01, format="%.2f", key="b6"))
        spec_b.discount_horizon_months = int(b1.number_input("B discount months", value=int(spec_b.discount_horizon_months), min_value=0, max_value=36, step=1, key="b7"))

    run_seed = int(train_seed) if train_seed else 42
    if st.button("Run Monte Carlo comparison", type="primary", key="run_mc"):
        with st.spinner("Simulating…"):
            res_a = mdl.simulate_intervention_distribution(cohort, spec_a, n_simulations=sim_runs, seed=run_seed + 3)
            res_b = mdl.simulate_intervention_distribution(cohort, spec_b, n_simulations=sim_runs, seed=run_seed + 5)
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
        qa.markdown(f"##### {names[0]}")
        qb.markdown(f"##### {names[1]}")
        qa.metric("Median net", mdl.format_currency(sum_a["median"]))
        qa.metric("5th–95th", f"{mdl.format_currency(sum_a['p05'])} → {mdl.format_currency(sum_a['p95'])}")
        qa.metric("P(net loss)", f"{sum_a['prob_negative']:.0%}")
        qb.metric("Median net", mdl.format_currency(sum_b["median"]))
        qb.metric("5th–95th", f"{mdl.format_currency(sum_b['p05'])} → {mdl.format_currency(sum_b['p95'])}")
        qb.metric("P(net loss)", f"{sum_b['prob_negative']:.0%}")
        st.plotly_chart(distribution_figure(res_a_state, res_b_state, names[0], names[1]), use_container_width=True)
        st.caption(
            "**Reading net impact:** the dashed line is **do nothing**. To the **right** means that Monte Carlo draw favored the "
            "intervention; to the **left** means intervention costs (fixed touch + any MRR discount) outweighed modeled "
            "churn-margin savings. Spread reflects **Tier A** uncertainty in lift and uptake, not re-sampled churn labels."
        )

        memo_ctx = {
            "cohort_accounts": int(cohort.shape[0]),
            "baseline_expected_margin_loss": round(baseline_loss, 2),
            "monte_carlo_runs": int(sim_runs),
            "scenario_a": {"name": names[0], "summary": {k: round(v, 4) for k, v in sum_a.items() if k != "label"}},
            "scenario_b": {"name": names[1], "summary": {k: round(v, 4) for k, v in sum_b.items() if k != "label"}},
        }
        st.session_state["memo_context"] = memo_ctx

        st.markdown("#### AI: interpret scenario outputs")
        if api_key:
            model_name = st.text_input("OpenAI model", value="gpt-4o-mini", key="openai_model_scen")
            if st.button("Explain scenario results with AI", key="explain_scen"):
                with st.spinner("Drafting…"):
                    try:
                        scen_obj = synthesize_memo(memo_ctx, api_key=api_key, model=model_name.strip())
                        st.subheader(scen_obj.get("headline", "Scenario readout"))
                        for bullet in scen_obj.get("bullets", []):
                            st.markdown(f"- {bullet}")
                        st.markdown("**Recommendation:** " + str(scen_obj.get("recommend", "neutral")))
                        for r in scen_obj.get("risks", []):
                            st.markdown(f"- {r}")
                        if scen_obj.get("confidence_note"):
                            st.caption(scen_obj["confidence_note"])
                        with st.expander("Metrics JSON sent to the model"):
                            st.code(json.dumps(memo_ctx, indent=2))
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
        else:
            st.caption("Add OpenAI key in secrets to narrate scenarios.")

        st.markdown("#### Structured decision memo (JSON)")
        if api_key:
            if st.button("Generate structured memo", key="memo_struct"):
                with st.spinner("Formatting memo…"):
                    try:
                        memo_model = str(st.session_state.get("openai_model_scen", "gpt-4o-mini")).strip() or "gpt-4o-mini"
                        memo_obj = synthesize_memo(memo_ctx, api_key=api_key, model=memo_model)
                        st.json(memo_obj)
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
    else:
        st.info("Run Monte Carlo to populate charts and AI scenario explanations.")


if __name__ == "__main__":
    main()
