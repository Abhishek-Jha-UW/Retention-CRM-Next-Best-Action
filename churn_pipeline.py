"""
Train churn models on user-supplied tabular data (upload / paste / sample).
Supports logistic regression, XGBoost, or automatic selection via holdout ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]


@dataclass
class TrainedChurnModel:
    """Fitted estimator + metadata for scoring and UI."""

    pipeline: Pipeline
    model_kind: str  # "logistic" | "xgboost"
    feature_columns: tuple[str, ...]
    target_column: str
    metrics: dict[str, Any]


def suggest_numeric_feature_columns(df: pd.DataFrame, target_col: str, exclude: list[str]) -> list[str]:
    """Numeric columns suitable as features (excluding target and id-like columns)."""
    exclude_set = {target_col, *exclude}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude_set]


def build_logistic_pipeline(feature_columns: tuple[str, ...]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(feature_columns))],
        remainder="drop",
        sparse_threshold=0.0,
    )
    clf = LogisticRegression(max_iter=8_000, class_weight="balanced", random_state=0)
    return Pipeline([("prep", pre), ("clf", clf)])


def build_xgboost_pipeline(feature_columns: tuple[str, ...], seed: int, pos_weight: float | None) -> Pipeline:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), list(feature_columns))],
        remainder="drop",
        sparse_threshold=0.0,
    )
    kwargs: dict[str, Any] = dict(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=seed % 10_000,
        n_jobs=-1,
        eval_metric="logloss",
    )
    if pos_weight is not None:
        kwargs["scale_pos_weight"] = pos_weight
    clf = XGBClassifier(**kwargs)
    return Pipeline([("prep", pre), ("clf", clf)])


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(threshold)}


def _roc_curve_dict(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}


def _fit_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_kind: str,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> TrainedChurnModel:
    pipeline.fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    auc = _roc_auc_safe(y_test, proba_test)
    metrics: dict[str, Any] = {
        "roc_auc_holdout": auc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_churn_rate": float(np.mean(y_test)),
        "confusion_matrix": _confusion_at_threshold(y_test, proba_test, 0.5),
        "roc_curve": _roc_curve_dict(y_test, proba_test),
    }
    if model_kind == "logistic":
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        names = prep.get_feature_names_out()
        coefs = clf.coef_[0]
        metrics["coefficients"] = {str(n): float(c) for n, c in zip(names, coefs)}
        metrics["intercept"] = float(clf.intercept_[0])
    elif model_kind == "xgboost":
        clf = pipeline.named_steps["clf"]
        imp = clf.feature_importances_
        metrics["feature_importance"] = {c: float(v) for c, v in zip(feature_columns, imp)}
    return TrainedChurnModel(
        pipeline=pipeline,
        model_kind=model_kind,
        feature_columns=feature_columns,
        target_column=target_column,
        metrics=metrics,
    )


def _auto_pick_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: tuple[str, ...],
    target_column: str,
    seed: int,
) -> TrainedChurnModel:
    """Pick logistic vs XGBoost by holdout ROC-AUC; prefer logistic on ties / small data."""
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    pos_weight = neg / pos

    log_pipe = build_logistic_pipeline(feature_columns)
    log_res = _fit_and_evaluate(log_pipe, X_train, y_train, X_test, y_test, "logistic", feature_columns, target_column)

    if XGBClassifier is None:
        return log_res

    xgb_pipe = build_xgboost_pipeline(feature_columns, seed, pos_weight)
    xgb_res = _fit_and_evaluate(xgb_pipe, X_train, y_train, X_test, y_test, "xgboost", feature_columns, target_column)

    a_log = log_res.metrics["roc_auc_holdout"]
    a_xgb = xgb_res.metrics["roc_auc_holdout"]
    if not np.isfinite(a_log) and not np.isfinite(a_xgb):
        return log_res
    if not np.isfinite(a_xgb):
        return log_res
    if not np.isfinite(a_log):
        return xgb_res
    if a_xgb > a_log + 0.01:
        xgb_res.metrics["auto_selection_note"] = f"Automatic chose XGBoost (AUC {a_xgb:.3f} vs logistic {a_log:.3f})."
        return xgb_res
    log_res.metrics["auto_selection_note"] = f"Automatic chose logistic regression (AUC {a_log:.3f} vs XGBoost {a_xgb:.3f})."
    return log_res


def train_churn_classifier(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_choice: str,
    test_size: float = 0.25,
    seed: int = 42,
) -> TrainedChurnModel:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    y_raw = df[target_column]
    if y_raw.dtype == object:
        y_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "churn": 1, "retain": 0}
        y = y_raw.astype(str).str.strip().str.lower().map(y_map)
        if y.isna().any():
            y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
    y = y.fillna(0).astype(int).values
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Target must be binary (0/1) after coercion.")

    feat_tuple = tuple(feature_columns)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    stratify = y if pos >= 30 and neg >= 30 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    choice = model_choice.strip().lower()
    if choice == "automatic":
        return _auto_pick_model(X_train, y_train, X_test, y_test, feat_tuple, target_column, seed)

    if choice in {"logistic", "logit", "logistic regression"}:
        pipe = build_logistic_pipeline(feat_tuple)
        return _fit_and_evaluate(pipe, X_train, y_train, X_test, y_test, "logistic", feat_tuple, target_column)

    if choice in {"xgboost", "xgb"}:
        pos_weight = neg / max(pos, 1)
        pipe = build_xgboost_pipeline(feat_tuple, seed, pos_weight)
        return _fit_and_evaluate(pipe, X_train, y_train, X_test, y_test, "xgboost", feat_tuple, target_column)

    raise ValueError(f"Unknown model choice: {model_choice}")


def score_full_dataframe(df: pd.DataFrame, trained: TrainedChurnModel) -> pd.DataFrame:
    """Append predicted churn probability column."""
    out = df.copy()
    X = out[list(trained.feature_columns)].apply(pd.to_numeric, errors="coerce")
    out["p_churn_horizon"] = trained.pipeline.predict_proba(X)[:, 1]
    return out


def build_template_csv() -> bytes:
    """Minimal CSV template aligned with the built-in sample schema."""
    header = (
        "account_id,tenure_months,login_events_30d,active_seats_pct,support_tickets_90d,"
        "nps_score,has_enterprise_addon,monthly_margin,mrr_monthly,margin_at_risk_horizon,churn_within_horizon\n"
        "A-1001,24,42,72,1,8,0,700,3500,2100,0\n"
        "A-1002,8,18,45,4,5,0,297,1500,890,1\n"
    )
    return header.encode("utf-8")
