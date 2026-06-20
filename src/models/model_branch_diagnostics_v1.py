from __future__ import annotations

import argparse
import html
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup
from src.models.model_utils import compute_auc, get_target_column, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
CB_LONG_OOF_PATH = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "long_oof_predictions.csv"
CB_LONG_RESULT_PATH = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "long_cv_results.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1"
OPTIONAL_SUB_DIR = OUT_DIR / "optional_submissions"
RANDOM_SEED = 42
N_SPLITS = 5
MISSING_CATEGORY = "__MISSING__"
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, target encoding, train+test concat, "
    "test-wide post-processing, and any use of test data for training or feature decisions were not used."
)

LGBM_CONFIG = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 100
CONFIG_NOTE = (
    "LightGBM diagnostics use 5-fold CV and native categorical handling fitted within each fold. "
    "n_estimators was set to 1500 with early stopping for runtime control."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model & Branch Diagnostics v1")
    parser.add_argument("--make-optional-submissions", action="store_true", help="Create optional LGBM/blend candidates only after OOF diagnostics.")
    parser.add_argument("--skip-frozen-specialist", action="store_true", help="Skip frozen LGBM specialist CV.")
    parser.add_argument("--quick", action="store_true", help="Use 3-fold and 800 estimators for a faster local smoke run.")
    return parser.parse_args()


def is_categorical_dtype(dtype: Any) -> bool:
    return dtype == "object" or str(dtype) in {"category", "bool", "boolean", "string", "str"}


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [col for col, dtype in X.dtypes.items() if is_categorical_dtype(dtype)]


def all_feature_columns(families: dict[str, list[str]], X: pd.DataFrame) -> list[str]:
    cols = all_cols_from_families(families, X)
    return [c for c in dict.fromkeys(cols) if c in X.columns]


def make_feature_sets(X: pd.DataFrame, families: dict[str, list[str]]) -> dict[str, list[str]]:
    all_cols = all_feature_columns(families, X)
    cat_cols = set(categorical_columns(X[all_cols]))
    engineered_keep_tokens = (
        "_bin",
        "_combo",
        "_pattern",
        "_branch",
        "_raw",
        "age_group_raw",
        "egg_source_raw",
        "sperm_source_raw",
    )
    raw_string_excludes = {
        "특정 시술 유형",
        "배아 생성 주요 이유",
        "시술 시기 코드",
        "시술 유형",
        "난자 출처",
        "정자 출처",
        "배란 유도 유형",
        "임신 시도 또는 마지막 임신 경과 연수",
        "난자 기증자 나이",
        "정자 기증자 나이",
    }
    numeric_binary = [c for c in all_cols if c not in cat_cols]
    structured = [
        c
        for c in all_cols
        if c not in raw_string_excludes and (c not in cat_cols or c.endswith(engineered_keep_tokens) or c in {"fresh_frozen_combo", "reason_branch", "specific_treatment_pattern"})
    ]
    return {
        "all_features_for_lgbm": all_cols,
        "numeric_binary_only": numeric_binary,
        "raw_plus_structured_flags": [c for c in dict.fromkeys(structured) if c in X.columns],
    }


def prepare_lgbm_fold(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = X_train.copy()
    valid = X_valid.copy()
    fit_cat_cols = [c for c in cat_cols if c in train.columns]
    for col in fit_cat_cols:
        tr = train[col].fillna(MISSING_CATEGORY).astype(str)
        va = valid[col].fillna(MISSING_CATEGORY).astype(str)
        levels = pd.Index(tr.unique()).append(pd.Index([MISSING_CATEGORY, "__UNKNOWN__"])).unique()
        va = va.where(va.isin(levels), "__UNKNOWN__")
        train[col] = pd.Categorical(tr, categories=levels)
        valid[col] = pd.Categorical(va, categories=levels)
    return train, valid, fit_cat_cols


def prepare_lgbm_full(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = X_train.copy()
    test = X_test.copy()
    fit_cat_cols = [c for c in cat_cols if c in train.columns]
    for col in fit_cat_cols:
        tr = train[col].fillna(MISSING_CATEGORY).astype(str)
        te = test[col].fillna(MISSING_CATEGORY).astype(str)
        levels = pd.Index(tr.unique()).append(pd.Index([MISSING_CATEGORY, "__UNKNOWN__"])).unique()
        te = te.where(te.isin(levels), "__UNKNOWN__")
        train[col] = pd.Categorical(tr, categories=levels)
        test[col] = pd.Categorical(te, categories=levels)
    return train, test, fit_cat_cols


def make_lgbm(config: dict[str, Any]):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**config)


def run_lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_name: str,
    feature_set: str,
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
    subset_mask: pd.Series | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in X.columns]
    if subset_mask is None:
        subset_mask = pd.Series(True, index=X.index)
    subset_mask = subset_mask.fillna(False)
    row_positions = np.flatnonzero(subset_mask.to_numpy())
    X_sub = X.iloc[row_positions][feature_cols].reset_index(drop=True)
    y_sub = y.iloc[row_positions].reset_index(drop=True)
    if len(row_positions) == 0 or y_sub.nunique() < 2:
        row = {
            "model_name": model_name,
            "model_type": "LightGBM",
            "feature_set": feature_set,
            "feature_count": len(feature_cols),
            "categorical_feature_count": 0,
            "numeric_feature_count": 0,
            "fold_auc_list": "",
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "oof_auc": np.nan,
            "best_iteration_list": "",
            "training_time_sec": 0.0,
            "note": "skipped: empty subset or one class",
        }
        return row, pd.DataFrame(), pd.DataFrame()

    min_class = int(y_sub.value_counts().min())
    folds = min(n_splits, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    cat_cols = categorical_columns(X_sub)
    num_cols = [c for c in X_sub.columns if c not in cat_cols]
    oof_parts: list[pd.DataFrame] = []
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    importances: list[pd.Series] = []
    start = time.time()

    from lightgbm import early_stopping, log_evaluation

    for fold, (tr, va) in enumerate(cv.split(X_sub, y_sub), start=1):
        X_tr, X_va = X_sub.iloc[tr], X_sub.iloc[va]
        y_tr, y_va = y_sub.iloc[tr], y_sub.iloc[va]
        X_tr, X_va, fit_cat_cols = prepare_lgbm_fold(X_tr, X_va, cat_cols)
        model = make_lgbm(config)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=fit_cat_cols,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(period=0)],
        )
        pred = model.predict_proba(X_va)[:, 1]
        auc = compute_auc(y_va, pred)
        fold_aucs.append(auc)
        best_iter = int(getattr(model, "best_iteration_", None) or config["n_estimators"])
        best_iterations.append(best_iter)
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_index": row_positions[va],
                    "ID": X.index[row_positions[va]].astype(str),
                    "y_true": y_va.to_numpy(),
                    "fold": fold,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "oof_pred": pred,
                }
            )
        )
        importances.append(pd.Series(model.feature_importances_, index=feature_cols, name=f"fold_{fold}"))

    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importances, axis=1)
    row = {
        "model_name": model_name,
        "model_type": "LightGBM",
        "feature_set": feature_set,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(cat_cols),
        "numeric_feature_count": len(num_cols),
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "training_time_sec": round(time.time() - start, 3),
        "note": CONFIG_NOTE,
    }
    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(c, "unknown") for c in imp.index],
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return row, oof, importance


def load_catboost_oof(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CB_LONG_OOF_PATH.exists():
        raise FileNotFoundError(f"Missing CatBoost OOF: {CB_LONG_OOF_PATH}")
    oof = pd.read_csv(CB_LONG_OOF_PATH)
    if "experiment" in oof.columns:
        oof = oof[oof["experiment"].eq("all_features_long_depth6")].copy()
    oof["model_name"] = "CB_long_all_features_depth6"
    oof["feature_set"] = "all_features_expanded_transfer"
    oof = oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]]
    if CB_LONG_RESULT_PATH.exists():
        result = pd.read_csv(CB_LONG_RESULT_PATH)
        result = result[result["experiment"].eq("all_features_long_depth6")].copy()
        row = {
            "model_name": "CB_long_all_features_depth6",
            "model_type": "CatBoost",
            "feature_set": "all_features_expanded_transfer",
            "feature_count": np.nan,
            "categorical_feature_count": np.nan,
            "numeric_feature_count": np.nan,
            "fold_auc_list": result["fold_auc_list"].iloc[0] if len(result) else "",
            "mean_auc": result["mean_auc"].iloc[0] if len(result) else np.nan,
            "std_auc": result["std_auc"].iloc[0] if len(result) else np.nan,
            "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
            "best_iteration_list": result["best_iteration_list"].iloc[0] if len(result) else "",
            "training_time_sec": 0.0,
            "note": f"Loaded from {CB_LONG_OOF_PATH}",
        }
    else:
        row = {
            "model_name": "CB_long_all_features_depth6",
            "model_type": "CatBoost",
            "feature_set": "all_features_expanded_transfer",
            "feature_count": np.nan,
            "categorical_feature_count": np.nan,
            "numeric_feature_count": np.nan,
            "fold_auc_list": "",
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
            "best_iteration_list": "",
            "training_time_sec": 0.0,
            "note": f"Loaded from {CB_LONG_OOF_PATH}",
        }
    return oof, pd.DataFrame([row])


def prediction_matrix(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.pivot_table(index="ID", columns="model_name", values="oof_pred")


def correlation_table(oof: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    rows = []
    models = list(mat.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pair = mat[[a, b]].dropna()
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n": len(pair),
                    "pearson": pair[a].corr(pair[b], method="pearson"),
                    "spearman": spearmanr(pair[a], pair[b]).correlation if len(pair) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_blends(oof: pd.DataFrame, cv_results: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    y = oof.drop_duplicates("ID").set_index("ID")["y_true"].loc[mat.index]
    cb_name = "CB_long_all_features_depth6"
    lgbm_rows = cv_results[
        cv_results["model_type"].eq("LightGBM")
        & ~cv_results["model_name"].str.contains("specialist", case=False, na=False)
    ].sort_values("oof_auc", ascending=False)
    if lgbm_rows.empty or cb_name not in mat.columns:
        return pd.DataFrame()
    best_lgbm = lgbm_rows.iloc[0]["model_name"]
    lowest_corr_lgbm = best_lgbm
    corr_to_cb = corr[(corr["model_a"].eq(cb_name)) | (corr["model_b"].eq(cb_name))].copy()
    if not corr_to_cb.empty:
        corr_to_cb["other"] = np.where(corr_to_cb["model_a"].eq(cb_name), corr_to_cb["model_b"], corr_to_cb["model_a"])
        corr_lgbm = corr_to_cb[corr_to_cb["other"].isin(lgbm_rows["model_name"])]
        if not corr_lgbm.empty:
            lowest_corr_lgbm = corr_lgbm.sort_values("pearson").iloc[0]["other"]
    base_pairs = [("best_lgbm", best_lgbm), ("lowest_corr_lgbm", lowest_corr_lgbm)]
    best_single_auc = cv_results["oof_auc"].max()
    rows = []
    for label, lgbm_name in base_pairs:
        if lgbm_name not in mat.columns:
            continue
        pair = mat[[cb_name, lgbm_name]].dropna()
        y_pair = y.loc[pair.index]
        pearson = pair[cb_name].corr(pair[lgbm_name], method="pearson")
        spearman = spearmanr(pair[cb_name], pair[lgbm_name]).correlation
        for cb_w, lgbm_w, blend_name in [
            (0.9, 0.1, "blend_cb_lgbm_90_10"),
            (0.8, 0.2, "blend_cb_lgbm_80_20"),
            (0.7, 0.3, "blend_cb_lgbm_70_30"),
            (0.6, 0.4, "blend_cb_lgbm_60_40"),
            (0.5, 0.5, "blend_equal_weight"),
        ]:
            pred = cb_w * pair[cb_name] + lgbm_w * pair[lgbm_name]
            rows.append(
                {
                    "blend_name": f"{blend_name}_{label}",
                    "base_models": f"{cb_name},{lgbm_name}",
                    "weights": f"{cb_w},{lgbm_w}",
                    "oof_auc": compute_auc(y_pair, pred),
                    "delta_vs_best_single": compute_auc(y_pair, pred) - best_single_auc,
                    "pearson_corr_between_base_models": pearson,
                    "spearman_corr_between_base_models": spearman,
                    "note": "OOF-only diagnostic blend; not a final ensemble decision",
                }
            )
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False)


def subgroup_specs(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    specs = [("overall", "all", pd.Series(True, index=frame.index))]
    for col, name in [("시술 유형", "treatment_type"), ("age_group_raw", "age_group"), ("egg_source_raw", "egg_source")]:
        if col in frame:
            for value in sorted(frame[col].dropna().astype(str).unique()):
                specs.append((name, value, frame[col].astype(str).eq(value)))
    for col in [
        "is_donor_egg",
        "is_own_egg",
        "is_fresh_embryo",
        "is_frozen_embryo",
        "embryo_transferred_flag",
        "no_embryo_transfer_flag",
        "transfer_day_5",
        "transfer_day_missing",
        "reason_current_treatment",
        "storage_only_flag",
        "donation_only_flag",
    ]:
        if col in frame:
            specs.append((col, "1", frame[col].eq(1)))
    return specs


def subgroup_auc_by_model(X: pd.DataFrame, raw: pd.DataFrame, oof: pd.DataFrame, blends: pd.DataFrame) -> pd.DataFrame:
    base = X.copy().reset_index(drop=True)
    base["ID"] = raw[ID_COLUMN].astype(str).to_numpy()
    y_map = oof.drop_duplicates("ID").set_index("ID")["y_true"]
    pred_wide = prediction_matrix(oof)
    extra_models: dict[str, pd.Series] = {}
    cb_name = "CB_long_all_features_depth6"
    for _, row in blends.iterrows():
        names = row["base_models"].split(",")
        weights = [float(x) for x in row["weights"].split(",")]
        if all(name in pred_wide.columns for name in names):
            extra_models[row["blend_name"]] = sum(w * pred_wide[name] for w, name in zip(weights, names))
    rows = []
    for model_name in list(pred_wide.columns) + list(extra_models.keys()):
        pred = pred_wide[model_name] if model_name in pred_wide.columns else extra_models[model_name]
        frame = base.merge(
            pd.DataFrame({"ID": pred.index.astype(str), "oof_pred": pred.to_numpy(), "y_true": y_map.loc[pred.index].to_numpy()}),
            on="ID",
            how="inner",
        ).dropna(subset=["oof_pred", "y_true"])
        for subgroup, value, mask in subgroup_specs(frame):
            sub = frame[mask.fillna(False)]
            if sub.empty:
                continue
            y_true = sub["y_true"]
            p = sub["oof_pred"]
            pos = p[y_true.eq(1)].mean() if y_true.eq(1).any() else np.nan
            neg = p[y_true.eq(0)].mean() if y_true.eq(0).any() else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "subgroup": subgroup,
                    "value": value,
                    "n": len(sub),
                    "positive_rate": y_true.mean(),
                    "auc": compute_auc(y_true, p) if y_true.nunique() > 1 else np.nan,
                    "pred_mean": p.mean(),
                    "calibration_gap": p.mean() - y_true.mean(),
                    "separation": pos - neg if pd.notna(pos) and pd.notna(neg) else np.nan,
                    "note": "" if y_true.nunique() > 1 else "only one class present",
                }
            )
    return pd.DataFrame(rows)


def branch_focus(subgroup: pd.DataFrame) -> pd.DataFrame:
    branch_map = {
        "overall": ("overall", "all"),
        "IVF": ("treatment_type", "IVF"),
        "DI": ("treatment_type", "DI"),
        "fresh": ("is_fresh_embryo", "1"),
        "frozen": ("is_frozen_embryo", "1"),
        "transfer_positive": ("embryo_transferred_flag", "1"),
        "no_transfer": ("no_embryo_transfer_flag", "1"),
        "day5_transfer": ("transfer_day_5", "1"),
        "donor_egg": ("is_donor_egg", "1"),
        "own_egg": ("is_own_egg", "1"),
        "current_treatment": ("reason_current_treatment", "1"),
        "storage_only": ("storage_only_flag", "1"),
    }
    rows = []
    cb_name = "CB_long_all_features_depth6"
    for branch, (subgroup_name, value) in branch_map.items():
        sub = subgroup[subgroup["subgroup"].eq(subgroup_name) & subgroup["value"].astype(str).eq(value)]
        if sub.empty:
            continue
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        cb = sub[sub["model_name"].eq(cb_name)]
        lgbm = sub[sub["model_name"].str.startswith("LGBM", na=False)]
        blend = sub[sub["model_name"].str.startswith("blend", na=False)]
        cb_auc = cb["auc"].iloc[0] if len(cb) else np.nan
        best_lgbm_auc = lgbm["auc"].max() if len(lgbm) else np.nan
        blend_auc = blend["auc"].max() if len(blend) else np.nan
        rows.append(
            {
                "branch_name": branch,
                "n": int(best["n"]),
                "positive_rate": best["positive_rate"],
                "best_model_by_auc": best["model_name"],
                "best_auc": best["auc"],
                "catboost_auc": cb_auc,
                "best_lgbm_auc": best_lgbm_auc,
                "delta_best_lgbm_vs_catboost": best_lgbm_auc - cb_auc if pd.notna(best_lgbm_auc) and pd.notna(cb_auc) else np.nan,
                "blend_auc": blend_auc,
                "delta_blend_vs_catboost": blend_auc - cb_auc if pd.notna(blend_auc) and pd.notna(cb_auc) else np.nan,
                "interpretation": "LGBM stronger" if pd.notna(best_lgbm_auc) and best_lgbm_auc > cb_auc else "CatBoost/blend competitive",
                "next_action": "Investigate branch-specific model" if branch in {"frozen", "transfer_positive", "day5_transfer", "DI", "donor_egg"} else "Monitor",
            }
        )
    return pd.DataFrame(rows)


def run_frozen_specialist(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_lgbm_cv(
        X,
        y,
        feature_cols,
        "frozen_LGBM_specialist",
        "all_features_for_lgbm",
        config,
        n_splits,
        lookup,
        subset_mask=X["is_frozen_embryo"].eq(1),
    )


def maybe_create_optional_submissions(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    cv_results: pd.DataFrame,
    blends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[Path], pd.DataFrame]:
    if not args.make_optional_submissions:
        return [], pd.DataFrame(
            [
                {
                    "candidate": "not_created",
                    "path": "",
                    "row_count_equals_sample": np.nan,
                    "id_order_matches_sample": np.nan,
                    "prediction_has_no_nan": np.nan,
                    "prediction_min": np.nan,
                    "prediction_max": np.nan,
                    "prediction_mean": np.nan,
                    "prediction_std": np.nan,
                    "duplicated_id_count": np.nan,
                    "note": "optional submissions disabled; diagnostics only",
                }
            ]
        )

    best_single = cv_results.sort_values("oof_auc", ascending=False).iloc[0]
    best_lgbm = cv_results[cv_results["model_type"].eq("LightGBM")].sort_values("oof_auc", ascending=False)
    best_blend = blends.sort_values("oof_auc", ascending=False).head(1)
    if best_lgbm.empty or best_lgbm.iloc[0]["oof_auc"] <= best_single["oof_auc"] and (best_blend.empty or best_blend.iloc[0]["delta_vs_best_single"] < 0.0003):
        return [], pd.DataFrame(
            [
                {
                    "candidate": "not_created",
                    "path": "",
                    "row_count_equals_sample": np.nan,
                    "id_order_matches_sample": np.nan,
                    "prediction_has_no_nan": np.nan,
                    "prediction_min": np.nan,
                    "prediction_max": np.nan,
                    "prediction_mean": np.nan,
                    "prediction_std": np.nan,
                    "duplicated_id_count": np.nan,
                    "note": "OOF conditions for optional candidate were not met",
                }
            ]
        )

    OPTIONAL_SUB_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = next((p for p in SAMPLE_SUBMISSION_CANDIDATES if p.exists()), None)
    sample = pd.read_csv(sample_path) if sample_path else pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(str), "probability": 0.0})
    pred_col = [c for c in sample.columns if c != ID_COLUMN][0]
    paths: list[Path] = []
    rows = []

    def fit_predict_lgbm(feature_set_name: str) -> np.ndarray:
        cols = feature_sets[feature_set_name]
        cat_cols = categorical_columns(X[cols])
        X_fit, X_pred, fit_cat_cols = prepare_lgbm_full(X[cols], X_test[cols], cat_cols)
        model = make_lgbm(config)
        model.fit(X_fit, y, categorical_feature=fit_cat_cols)
        return model.predict_proba(X_pred)[:, 1]

    lgbm_model_name = best_lgbm.iloc[0]["model_name"]
    lgbm_feature_set = best_lgbm.iloc[0]["feature_set"]
    lgbm_pred = fit_predict_lgbm(lgbm_feature_set)
    lgbm_path = OPTIONAL_SUB_DIR / "candidate_lgbm_all_features.csv"
    lgbm_sub = sample.copy()
    lgbm_sub[pred_col] = lgbm_pred
    save_table(lgbm_path, lgbm_sub)
    paths.append(lgbm_path)
    rows.append(sanity_row("candidate_lgbm_all_features", lgbm_path, lgbm_sub, sample))

    if not best_blend.empty:
        weights = [float(x) for x in best_blend.iloc[0]["weights"].split(",")]
        # CatBoost test prediction is intentionally not regenerated here to avoid another long run.
        # The blend candidate is only created when a CatBoost candidate from the long stage exists.
        cb_candidate = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "submissions" / "candidate_catboost_long_all_features.csv"
        if cb_candidate.exists():
            cb_sub = pd.read_csv(cb_candidate)
            blend_sub = sample.copy()
            blend_sub[pred_col] = weights[0] * cb_sub[pred_col].to_numpy() + weights[1] * lgbm_pred
            blend_path = OPTIONAL_SUB_DIR / "candidate_cb_lgbm_diagnostic_blend.csv"
            save_table(blend_path, blend_sub)
            paths.append(blend_path)
            rows.append(sanity_row("candidate_cb_lgbm_diagnostic_blend", blend_path, blend_sub, sample))
    return paths, pd.DataFrame(rows)


def sanity_row(candidate: str, path: Path, sub: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    pred_col = [c for c in sub.columns if c != ID_COLUMN][0]
    pred = sub[pred_col]
    return {
        "candidate": candidate,
        "path": str(path),
        "row_count_equals_sample": len(sub) == len(sample),
        "id_order_matches_sample": sub[ID_COLUMN].astype(str).equals(sample[ID_COLUMN].astype(str)),
        "prediction_has_no_nan": pred.notna().all(),
        "prediction_min": pred.min(),
        "prediction_max": pred.max(),
        "prediction_mean": pred.mean(),
        "prediction_std": pred.std(),
        "duplicated_id_count": int(sub[ID_COLUMN].duplicated().sum()),
        "note": "candidate only, not submitted",
    }


def build_report(tables: dict[str, pd.DataFrame], optional_paths: list[Path]) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def t(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    cv = tables["model_cv_results"].sort_values("oof_auc", ascending=False)
    best = cv.iloc[0]
    html_parts = [
        f"<style>{css}</style>",
        "<h1>Model & Branch Diagnostics v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best single model: {html.escape(str(best['model_name']))}, OOF AUC {best['oof_auc']:.6f}. This is diagnostics, not final ensemble selection.</p>",
        "<h2>Leakage and Prohibited Methods</h2>",
        "<p>No pseudo-labeling, self-training, target encoding, train+test concat, test-wide post-processing, or automatic submission was used.</p>",
        "<h2>Why this is Model Diagnostics, not Final Ensemble</h2>",
        "<p>Blend weights are evaluated only on OOF predictions to detect possible diversity. They are not public-LB tuned.</p>",
        "<h2>CatBoost Baseline Recap</h2>", t("model_cv_results", 10),
        "<h2>LightGBM Model Comparison</h2>", t("model_cv_results", 30),
        "<h2>OOF Correlation</h2>", t("model_oof_correlation", 30),
        "<h2>Diagnostic Blend Results</h2>", t("diagnostic_blend_results", 30),
        "<h2>Subgroup AUC by Model</h2>", t("subgroup_auc_by_model", 50),
        "<h2>Branch Focus Diagnostics</h2>", t("branch_focus_diagnostics", 30),
        "<h2>Frozen Specialist Diagnostics</h2>", t("branch_focus_diagnostics", 20),
        "<h2>Feature Importance</h2>", t("model_feature_importance", 50),
        "<h2>Optional Submission Candidates</h2>", t("optional_submission_sanity", 10),
        "<h2>Interpretation</h2>",
        "<p>Use this report to decide whether LightGBM adds ranking diversity and whether frozen/transfer/day5 branches need separate modeling.</p>",
        "<h2>Next Step Recommendation</h2>",
        "<p>If OOF blend gain is stable and correlations are below near-duplicate levels, run a separate controlled ensemble stage. Otherwise continue branch feature work.</p>",
    ]
    if optional_paths:
        html_parts.append("<ul>" + "".join(f"<li>{html.escape(str(p))}</li>" for p in optional_paths) + "</ul>")
    (OUT_DIR / "model_branch_diagnostics_report.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []
    n_splits = 3 if args.quick else N_SPLITS
    config = dict(LGBM_CONFIG)
    if args.quick:
        config["n_estimators"] = 800

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        train = pd.read_csv(TRAIN_PATH)
        target = get_target_column(train)
        y = train[target].astype(int)
        X = make_art_features(train)
        X.index = train[ID_COLUMN].astype(str)
        families = save_feature_families(OUT_DIR / "feature_families.json", X.columns.tolist())
        lookup = family_lookup(families)
        feature_sets = make_feature_sets(X, families)
        save_json(
            OUT_DIR / "model_config.json",
            {
                "cv": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
                "lgbm": config,
                "catboost_oof_path": str(CB_LONG_OOF_PATH),
                "note": CONFIG_NOTE,
            },
        )

        oof_frames: list[pd.DataFrame] = []
        result_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []

        cb_oof, cb_result = load_catboost_oof(train)
        oof_frames.append(cb_oof)
        result_frames.append(cb_result)
        completed.append("CB_long_all_features_depth6_loaded")

        lgbm_specs = [
            ("LGBM_A_all_features", "all_features_for_lgbm"),
            ("LGBM_B_numeric_binary_only", "numeric_binary_only"),
            ("LGBM_C_structured_flags", "raw_plus_structured_flags"),
        ]
        for model_name, feature_set in lgbm_specs:
            row, oof, imp = run_lgbm_cv(X, y, feature_sets[feature_set], model_name, feature_set, config, n_splits, lookup)
            result_frames.append(pd.DataFrame([row]))
            if not oof.empty:
                oof_frames.append(oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]])
                importance_frames.append(imp)
            completed.append(model_name)

        branch_importance = pd.DataFrame()
        if not args.skip_frozen_specialist:
            row, oof, imp = run_frozen_specialist(X, y, feature_sets["all_features_for_lgbm"], config, n_splits, lookup)
            result_frames.append(pd.DataFrame([row]))
            if not oof.empty:
                oof_frames.append(oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]])
                branch_importance = imp
            completed.append("frozen_LGBM_specialist")
        else:
            skipped.append("frozen_LGBM_specialist: skipped by flag")

        oof_all = pd.concat(oof_frames, ignore_index=True)
        cv_results = pd.concat(result_frames, ignore_index=True).sort_values("oof_auc", ascending=False)
        importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        corr = correlation_table(oof_all)
        blends = make_blends(oof_all, cv_results, corr)
        subgroup = subgroup_auc_by_model(X, train, oof_all, blends)
        branch = branch_focus(subgroup)

        optional_paths: list[Path] = []
        optional_sanity = pd.DataFrame()
        if args.make_optional_submissions:
            test = pd.read_csv(TEST_PATH)
            X_test = make_art_features(test)
            X_test.index = test[ID_COLUMN].astype(str)
            optional_paths, optional_sanity = maybe_create_optional_submissions(args, train, test, X, X_test, y, feature_sets, cv_results, blends, config)
        else:
            optional_sanity = pd.DataFrame(
                [
                    {
                        "candidate": "not_created",
                        "path": "",
                        "row_count_equals_sample": np.nan,
                        "id_order_matches_sample": np.nan,
                        "prediction_has_no_nan": np.nan,
                        "prediction_min": np.nan,
                        "prediction_max": np.nan,
                        "prediction_mean": np.nan,
                        "prediction_std": np.nan,
                        "duplicated_id_count": np.nan,
                        "note": "optional submissions disabled; diagnostics only",
                    }
                ]
            )

        tables = {
            "model_cv_results": cv_results,
            "model_oof_predictions": oof_all,
            "model_oof_correlation": corr,
            "diagnostic_blend_results": blends,
            "subgroup_auc_by_model": subgroup,
            "branch_focus_diagnostics": branch,
            "model_feature_importance": importance,
            "branch_feature_importance": branch_importance,
            "optional_submission_sanity": optional_sanity,
        }
        output_name_map = {
            "model_cv_results": "model_cv_results.csv",
            "model_oof_predictions": "model_oof_predictions.csv",
            "model_oof_correlation": "model_oof_correlation.csv",
            "diagnostic_blend_results": "diagnostic_blend_results.csv",
            "subgroup_auc_by_model": "subgroup_auc_by_model.csv",
            "branch_focus_diagnostics": "branch_focus_diagnostics.csv",
            "model_feature_importance": "model_feature_importance.csv",
            "branch_feature_importance": "branch_feature_importance.csv",
            "optional_submission_sanity": "optional_submission_sanity.csv",
        }
        for key, filename in output_name_map.items():
            save_table(OUT_DIR / filename, tables[key])
        build_report(tables, optional_paths)

        warnings_list.extend(str(w.message) for w in caught)
        import lightgbm
        import sklearn

        best_single = cv_results.iloc[0]
        best_lgbm = cv_results[cv_results["model_type"].eq("LightGBM")].sort_values("oof_auc", ascending=False).head(1)
        best_blend = blends.sort_values("oof_auc", ascending=False).head(1) if not blends.empty else pd.DataFrame()
        key_branch = branch[branch["branch_name"].isin(["frozen", "transfer_positive", "day5_transfer", "DI", "donor_egg"])].to_dict(orient="records")
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"sklearn_version: {sklearn.__version__}",
            f"lightgbm_version: {lightgbm.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"test_path: {TEST_PATH}",
            f"target_column: {target}",
            f"cv_setting: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
            f"catboost_setting: loaded from {CB_LONG_OOF_PATH}",
            f"lightgbm_setting: {config}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"best_single_model: {best_single['model_name']}",
            f"best_single_oof_auc: {best_single['oof_auc']:.6f}",
            f"best_lightgbm: {best_lgbm.iloc[0]['model_name'] if len(best_lgbm) else 'none'}",
            f"best_lightgbm_oof_auc: {best_lgbm.iloc[0]['oof_auc'] if len(best_lgbm) else np.nan}",
            f"best_diagnostic_blend: {best_blend.iloc[0]['blend_name'] if len(best_blend) else 'none'}",
            f"best_diagnostic_blend_oof_auc: {best_blend.iloc[0]['oof_auc'] if len(best_blend) else np.nan}",
            f"key_branch_findings: {key_branch}",
            f"optional_candidate_paths: {','.join(str(p) for p in optional_paths) if optional_paths else 'none'}",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "model_branch_diagnostics_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
