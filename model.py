"""
Decision Intelligence Simulator — core modeling + Monte Carlo simulation.

Synthetic data generator has a documented DGP so offline evaluation is honest.
Tier A causal lift enters as user-defined uncertainty ranges (sensitivity analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "tenure_months",
    "login_events_30d",
    "active_seats_pct",
    "support_tickets_90d",
    "nps_score",
    "has_enterprise_addon",
)


@dataclass
class InterventionSpec:
    """Tier A assumptions: uplift and uptake are plausible ranges (not causal truth)."""

    name: str
    uplift_pp_low: float = 0.02
    uplift_pp_high: float = 0.06
    uptake_low: float = 0.45
    uptake_high: float = 0.70
    fixed_cost_per_touched_account: float = 0.0
    discount_rate_on_mrr: float = 0.0  # fraction of monthly revenue given up for acceptors
    discount_horizon_months: int = 0  # e.g., 12 for year-long concession

    def validate(self) -> None:
        if self.uplift_pp_low > self.uplift_pp_high or self.uptake_low > self.uptake_high:
            raise ValueError("Lift/uptake ranges must be low <= high.")
        if any(x < 0 for x in (self.uplift_pp_low, self.discount_rate_on_mrr)):
            raise ValueError("Ranges must be non-negative where applicable.")


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)
    metrics: dict | None = None


def generate_synthetic_crm_data(
    n_customers: int = 4_000,
    seed: int = 42,
    renewal_horizon_months: int = 3,
) -> pd.DataFrame:
    """
    Documented DGP: churn is a noisy logistic function of engagement + tenure + friction.

    Intended for demos: you can sanity-check metrics on *held-out synthetic* data —
    optimism is bounded because we did not peek at test labels during training.
    """
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n_customers)

    baseline_usage = rng.normal(55, 12, size=n_customers).clip(15, 100)
    tenure_boost = np.sqrt(np.asarray(tenure_months, dtype=float)) * 2.6
    login_events_30d = rng.poisson(np.clip(baseline_usage + tenure_boost - 65, 1, None)).clip(1, None)

    has_enterprise_addon = rng.binomial(1, 0.18 + np.clip(tenure_months / 120, 0, 0.15), size=n_customers)

    breadth = rng.beta(7, 2, size=n_customers)
    seats_mult = rng.uniform(40, 100, size=n_customers)
    active_seats_pct = np.clip(breadth * seats_mult / 110 * 85 + rng.normal(0, 10, size=n_customers), 12, 100)

    friction = rng.exponential(scale=2.25, size=n_customers).clip(0, 48)
    support_tickets_90d = rng.poisson(friction)

    base_nps = 6.75 + breadth * 1.95 - friction * 0.08 + has_enterprise_addon * 0.45
    nps_score = np.clip(np.round(base_nps + rng.normal(0, 1.05, size=n_customers)), 1, 10)

    logits = np.zeros(n_customers, dtype=float)
    logits += -5.35
    logits += np.log(login_events_30d + 5.5) * -0.93
    logits += tenure_months * -0.028
    logits += np.log(active_seats_pct + 15.5) * -0.014
    logits += support_tickets_90d * 0.15
    logits += (np.asarray(nps_score, dtype=float) - 6.85) * -0.26
    logits += has_enterprise_addon * -0.42
    # Demo-friendly DGP: moderate noise + stronger standardized signal → holdout AUC typically mid‑0.7s+ on this feature set.
    logits += rng.normal(0, 0.22, size=n_customers)

    logits = (logits - logits.mean()) / (logits.std() + 1e-6) * 1.28 - 1.92

    churn_probability = 1 / (1 + np.exp(-logits))
    churn_within_horizon = rng.binomial(1, churn_probability)

    mrr_monthly = np.clip(rng.normal(1200 + has_enterprise_addon * 760, 620), 120.0, None)
    blended_margin_frac = rng.uniform(0.42, 0.63, size=n_customers)
    monthly_margin = mrr_monthly * blended_margin_frac

    margin_at_horizon_if_churn_saves = monthly_margin * float(renewal_horizon_months)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "login_events_30d": login_events_30d,
            "active_seats_pct": active_seats_pct,
            "support_tickets_90d": support_tickets_90d.astype(int),
            "nps_score": nps_score.astype(int),
            "has_enterprise_addon": has_enterprise_addon.astype(np.int64),
            "monthly_margin": monthly_margin,
            "renewal_horizon_months": int(renewal_horizon_months),
            "margin_at_risk_horizon": margin_at_horizon_if_churn_saves,
            "mrr_monthly": mrr_monthly,
            "churn_within_horizon": churn_within_horizon.astype(np.int64),
        }
    )
    df["renewal_horizon_months"] = int(renewal_horizon_months)
    df["segment"] = pd.cut(df["monthly_margin"], bins=[0, 600, 1500, 10_000], labels=["SMB", "Mid-market", "Strategic"]).astype(str)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("scalar", StandardScaler(), list(FEATURE_COLUMNS))],
        remainder="drop",
        sparse_threshold=0.0,
    )
    logistic = LogisticRegression(max_iter=5_000, class_weight="balanced", random_state=0)
    return Pipeline([("prep", preprocessor), ("clf", logistic)])


def train_churn_model(df: pd.DataFrame, test_size: float = 0.25, seed: int = 7) -> ModelBundle:
    """Train interpretable baseline (scaled logistic regression)."""
    X = df[list(FEATURE_COLUMNS)]
    y = df["churn_within_horizon"].values
    pos = int(y.sum())
    neg = int(len(y) - pos)
    stratify = y if pos >= 40 and neg >= 40 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) >= 2:
        auc = float(roc_auc_score(y_test, proba_test))
    else:
        auc = float("nan")
    metrics = {
        "roc_auc": auc,
        "test_pos_rate": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    try:
        prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=8, strategy="quantile")
        metrics["calibration"] = {
            "prob_pred": prob_pred.tolist(),
            "prob_true": prob_true.tolist(),
        }
    except ValueError:
        metrics["calibration"] = None

    coeffs = pipeline.named_steps["clf"].coef_[0]
    metrics["logistic_coef_scaled"] = {col: float(c) for col, c in zip(FEATURE_COLUMNS, coeffs)}
    metrics["logistic_intercept_scaled"] = float(pipeline.named_steps["clf"].intercept_[0])
    bundle = ModelBundle(pipeline=pipeline, metrics=metrics)
    return bundle


def score_accounts(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = df.copy()
    probs = bundle.pipeline.predict_proba(out[list(FEATURE_COLUMNS)])[:, 1]
    out["p_churn_horizon"] = probs
    out["expected_margin_loss_horizon"] = out["margin_at_risk_horizon"].to_numpy(dtype=float) * out["p_churn_horizon"].to_numpy(dtype=float)
    try:
        out["risk_tier"] = pd.qcut(
            out["expected_margin_loss_horizon"], q=4, labels=["Low", "Medium", "Elevated", "Critical"], duplicates="drop"
        )
    except ValueError:
        out["risk_tier"] = "Unbucketed"
    return out


def prioritize_cohort(df: pd.DataFrame, top_n: int, segment_filter: tuple[str, ...] | None = None) -> pd.DataFrame:
    cand = df
    if segment_filter:
        cand = cand[cand["segment"].isin(segment_filter)]
    return cand.nlargest(top_n, "expected_margin_loss_horizon").copy()


def simulate_intervention_distribution(
    cohort: pd.DataFrame,
    spec: InterventionSpec,
    n_simulations: int = 400,
    seed: int = 11,
) -> dict:
    """
    Monte Carlo over uncertain lift + uptake (Tier A). Returns net margin vs do-nothing for touched accounts.
    Expected churn reductions follow: p_new = clip(p - uplift_pp * uptake, eps, 1-eps).

    uplift_pp is sampled in percentage points toward non-churn probability mass (risk reduction):
    interpreting as additive risk reduction proxy on logit is avoided for auditability in v1 —
    additive on probability with clipping is deliberate for portfolio clarity.
    """
    spec.validate()
    rng = np.random.default_rng(seed)

    p = cohort["p_churn_horizon"].to_numpy(dtype=float)
    margin = cohort["margin_at_risk_horizon"].to_numpy(dtype=float)
    mrr = cohort["mrr_monthly"].to_numpy(dtype=float)

    n = cohort.shape[0]
    uplift_samples = rng.uniform(spec.uplift_pp_low, spec.uplift_pp_high, size=(n_simulations, 1))
    uptake_samples = rng.uniform(spec.uptake_low, spec.uptake_high, size=(n_simulations, 1))
    effective_pp = uplift_samples * uptake_samples

    p_row = np.broadcast_to(p, (n_simulations, n))
    margin_row = np.broadcast_to(margin, (n_simulations, n))
    new_p = np.clip(p_row - effective_pp, 1e-6, 1.0 - 1e-6)

    baseline_expected_loss_rows = p_row * margin_row
    post_expected_loss_rows = new_p * margin_row
    churn_margin_saved_rows = baseline_expected_loss_rows - post_expected_loss_rows

    fixed_each = np.full((n_simulations, n), spec.fixed_cost_per_touched_account, dtype=float)

    uptake_row = np.broadcast_to(uptake_samples, (n_simulations, n))

    discount_cost_rows = np.zeros((n_simulations, n), dtype=float)
    if spec.discount_rate_on_mrr > 0 and spec.discount_horizon_months > 0:
        mrr_row = np.broadcast_to(mrr, (n_simulations, n))
        dollar_discount_per_accept = spec.discount_rate_on_mrr * mrr_row * float(spec.discount_horizon_months)
        discount_cost_rows = uptake_row * dollar_discount_per_accept

    gross_benefit = churn_margin_saved_rows.sum(axis=1)
    intervention_fixed_total = fixed_each.sum(axis=1)
    discount_totals = discount_cost_rows.sum(axis=1)
    net_benefit = gross_benefit - intervention_fixed_total - discount_totals

    return {
        "net_benefit": net_benefit,
        "gross_churn_margin_saved": gross_benefit,
        "fixed_cost": intervention_fixed_total,
        "discount_cost": discount_totals,
        "baseline_expected_loss_total": baseline_expected_loss_rows.sum(axis=1),
        "post_expected_loss_total": post_expected_loss_rows.sum(axis=1),
        "n_accounts": int(n),
    }


def summarize_draws(draws: np.ndarray, label: str) -> dict:
    arr = np.asarray(draws, dtype=float)
    return {
        "label": label,
        "median": float(np.percentile(arr, 50)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "prob_negative": float(np.mean(arr < 0)),
    }


def format_currency(x: float) -> str:
    """Human-readable USD (plain text)."""
    abs_x = abs(x)
    return f"-${abs_x:,.0f}" if x < 0 else f"${abs_x:,.0f}"


def attach_simulation_economics(
    df: pd.DataFrame,
    *,
    margin_column: str,
    margin_is_monthly: bool,
    horizon_months: int,
    mrr_column: str | None,
    segment_column: str | None,
) -> pd.DataFrame:
    """
    Normalize user-uploaded columns into the names expected by prioritize_cohort
    and simulate_intervention_distribution.
    """
    if "p_churn_horizon" not in df.columns:
        raise ValueError("Expected column p_churn_horizon from churn scoring step.")
    if margin_column not in df.columns:
        raise ValueError(f"Margin column '{margin_column}' not found.")

    out = df.copy()
    margin_src = pd.to_numeric(out[margin_column], errors="coerce").fillna(0.0).astype(float)
    if margin_is_monthly:
        out["margin_at_risk_horizon"] = margin_src * float(max(horizon_months, 1))
    else:
        out["margin_at_risk_horizon"] = margin_src

    if mrr_column and mrr_column in out.columns:
        out["mrr_monthly"] = pd.to_numeric(out[mrr_column], errors="coerce").fillna(0.0).astype(float).clip(lower=1.0)
    else:
        out["mrr_monthly"] = (out["margin_at_risk_horizon"] / float(max(horizon_months, 1))).clip(lower=1.0)

    if segment_column and segment_column in out.columns:
        out["segment"] = out[segment_column].astype(str)
    else:
        out["segment"] = "All"

    out["renewal_horizon_months"] = int(horizon_months)
    out["expected_margin_loss_horizon"] = out["p_churn_horizon"].astype(float) * out["margin_at_risk_horizon"].astype(float)
    try:
        out["risk_tier"] = pd.qcut(
            out["expected_margin_loss_horizon"],
            q=4,
            labels=["Low", "Medium", "Elevated", "Critical"],
            duplicates="drop",
        )
    except ValueError:
        out["risk_tier"] = "Unbucketed"
    return out

