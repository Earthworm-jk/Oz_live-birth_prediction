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
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup
from src.models.model_branch_diagnostics_v1 import load_catboost_oof, subgroup_auc_by_model
from src.models.model_utils import compute_auc, get_categorical_columns, get_target_column, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
LGBM_OOF_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_oof_predictions.csv"
LGBM_RESULT_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_cv_results.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "model_zoo_v1"
OPTIONAL_SUB_DIR = OUT_DIR / "optional_submissions"
RANDOM_SEED = 42
N_SPLITS = 5
CB_NAME = "CB_long_all_features_depth6"
LGBM_A_NAME = "LGBM_A_all_features"
PREVIOUS_BEST_BLEND_AUC = 0.740462
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, target encoding, train+test concat, test-wide post-processing, "
    "and any use of test data for training or feature decisions were not used."
)


XGB_A_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 150,
}
XGB_C_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 40,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 8.0,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 150,
}
ET_CONFIG = {
    "n_estimators": 800,
    "max_depth": None,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "bootstrap": False,
    "class_weight": None,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
RF_CONFIG = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
LOGIT_CONFIG = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "saga",
    "max_iter": 3000,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Zoo v1")
    parser.add_argument("--quick", action="store_true", help="Use 3-fold CV and reduced model sizes.")
    parser.add_argument("--include-optional", action="store_true", help="Also run shallow XGB and RandomForest optional models.")
    parser.add_argument("--make-optional-submissions", action="store_true", help="Create optional candidates only if OOF criteria are met.")
    return parser.parse_args()


def unique_existing(cols: list[str], X: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in X.columns]


def make_feature_sets(X: pd.DataFrame, families: dict[str, list[str]]) -> dict[str, list[str]]:
    all_cols = unique_existing(all_cols_from_families(families, X), X)
    cat_cols = set(get_categorical_columns(X[all_cols]))
    structured_numeric = [c for c in all_cols if c not in cat_cols]
    low_medium_cat = [
        c
        for c in all_cols
        if c in cat_cols
        and X[c].nunique(dropna=True) <= 40
        and not c.endswith("_raw_normalized")
        and "normalized" not in c
    ]
    return {
        "structured_numeric": structured_numeric,
        "ohe_light_categorical": structured_numeric + low_medium_cat,
        "raw_catboost_reference": all_cols,
    }


def numeric_and_categorical(X: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    cat_cols = get_categorical_columns(X[cols])
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols, cat_cols = numeric_and_categorical(X, feature_cols)
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.3), num_cols, cat_cols


def encoded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" or trans == "drop":
                continue
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                try:
                    names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    names.extend(cols)
        return names


def original_feature_from_encoded(encoded: str) -> str:
    if encoded.startswith("num__"):
        return encoded.replace("num__", "", 1)
    if encoded.startswith("cat__"):
        rest = encoded.replace("cat__", "", 1)
        return rest.split("_", 1)[0]
    return encoded


def load_lgbm_a_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LGBM_OOF_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    oof = pd.read_csv(LGBM_OOF_PATH)
    oof = oof[oof["model_name"].eq(LGBM_A_NAME)].copy()
    if oof.empty:
        return pd.DataFrame(), pd.DataFrame()
    oof = oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]]
    result_row = {
        "model_name": LGBM_A_NAME,
        "model_type": "LightGBM",
        "feature_set": "all_features_for_lgbm",
        "feature_count": np.nan,
        "encoded_feature_count": np.nan,
        "fold_auc_list": "",
        "mean_auc": np.nan,
        "std_auc": np.nan,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": "",
        "training_time_sec": 0.0,
        "delta_vs_catboost_baseline": np.nan,
        "note": f"Loaded from {LGBM_OOF_PATH}",
    }
    if LGBM_RESULT_PATH.exists():
        res = pd.read_csv(LGBM_RESULT_PATH)
        res = res[res["model_name"].eq(LGBM_A_NAME)]
        if not res.empty:
            first = res.iloc[0]
            result_row.update(
                {
                    "feature_count": first.get("feature_count", np.nan),
                    "encoded_feature_count": first.get("feature_count", np.nan),
                    "fold_auc_list": first.get("fold_auc_list", ""),
                    "mean_auc": first.get("mean_auc", np.nan),
                    "std_auc": first.get("std_auc", np.nan),
                    "best_iteration_list": first.get("best_iteration_list", ""),
                }
            )
    return oof, pd.DataFrame([result_row])


def run_xgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_name: str,
    feature_set: str,
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    from xgboost import XGBClassifier

    feature_cols = unique_existing(feature_cols, X)
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.Series] = []
    encoded_count = np.nan
    start = time.time()
    for fold, (tr, va) in enumerate(cv.split(X[feature_cols], y), start=1):
        prep, _, _ = make_preprocessor(X.iloc[tr], feature_cols)
        X_tr = prep.fit_transform(X.iloc[tr][feature_cols])
        X_va = prep.transform(X.iloc[va][feature_cols])
        model = XGBClassifier(**config)
        model.fit(X_tr, y.iloc[tr], eval_set=[(X_va, y.iloc[va])], verbose=False)
        pred = model.predict_proba(X_va)[:, 1]
        auc = compute_auc(y.iloc[va], pred)
        fold_aucs.append(auc)
        best_iter = int(getattr(model, "best_iteration", config["n_estimators"]))
        best_iterations.append(best_iter)
        encoded_names = encoded_feature_names(prep)
        encoded_count = len(encoded_names)
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        values = pd.Series(0.0, index=encoded_names)
        for key, val in score.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(encoded_names):
                    values.iloc[idx] = float(val)
        importance_parts.append(values.rename(f"fold_{fold}"))
        oof_parts.append(
            pd.DataFrame(
                {
                    "ID": X.index[va].astype(str),
                    "y_true": y.iloc[va].to_numpy(),
                    "fold": fold,
                    "model_name": model_name,
                    "model_type": "XGBoost",
                    "feature_set": feature_set,
                    "oof_pred": pred,
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importance_parts, axis=1)
    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": [original_feature_from_encoded(c) for c in imp.index],
            "encoded_feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(original_feature_from_encoded(c), "encoded_or_unknown") for c in imp.index],
            "note": "xgboost gain importance",
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    row = cv_row(model_name, "XGBoost", feature_set, len(feature_cols), encoded_count, fold_aucs, best_iterations, oof, time.time() - start, "XGBoost CV with fold-local preprocessing")
    return row, oof, importance


def run_tree_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_name: str,
    model_type: str,
    feature_set: str,
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model_cls = ExtraTreesClassifier if model_type == "ExtraTrees" else RandomForestClassifier
    feature_cols = unique_existing(feature_cols, X)
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    oof_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.Series] = []
    encoded_count = len(feature_cols)
    start = time.time()
    for fold, (tr, va) in enumerate(cv.split(X[feature_cols], y), start=1):
        prep = SimpleImputer(strategy="median")
        X_tr = prep.fit_transform(X.iloc[tr][feature_cols])
        X_va = prep.transform(X.iloc[va][feature_cols])
        model = model_cls(**config)
        model.fit(X_tr, y.iloc[tr])
        pred = model.predict_proba(X_va)[:, 1]
        fold_aucs.append(compute_auc(y.iloc[va], pred))
        importance_parts.append(pd.Series(model.feature_importances_, index=feature_cols, name=f"fold_{fold}"))
        oof_parts.append(
            pd.DataFrame(
                {
                    "ID": X.index[va].astype(str),
                    "y_true": y.iloc[va].to_numpy(),
                    "fold": fold,
                    "model_name": model_name,
                    "model_type": model_type,
                    "feature_set": feature_set,
                    "oof_pred": pred,
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importance_parts, axis=1)
    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": imp.index,
            "encoded_feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(c, "unknown") for c in imp.index],
            "note": f"{model_type} feature_importances_",
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    row = cv_row(model_name, model_type, feature_set, len(feature_cols), encoded_count, fold_aucs, [], oof, time.time() - start, f"{model_type} CV with fold-local imputer")
    return row, oof, importance


def run_logit_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model_name = "RidgeLogit_ohe_light"
    feature_set = "ohe_light_categorical"
    feature_cols = unique_existing(feature_cols, X)
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    oof_parts: list[pd.DataFrame] = []
    coef_parts: list[pd.Series] = []
    encoded_count = np.nan
    start = time.time()
    use_sgd = bool(config.pop("_use_sgd_quick", False))
    for fold, (tr, va) in enumerate(cv.split(X[feature_cols], y), start=1):
        prep, _, _ = make_preprocessor(X.iloc[tr], feature_cols)
        X_tr = prep.fit_transform(X.iloc[tr][feature_cols])
        X_va = prep.transform(X.iloc[va][feature_cols])
        scaler = StandardScaler(with_mean=False)
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        if use_sgd:
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.0005,
                max_iter=int(config.get("max_iter", 200)),
                tol=1e-3,
                n_jobs=config.get("n_jobs", -1),
                random_state=RANDOM_SEED,
            )
        else:
            model = LogisticRegression(**config)
        model.fit(X_tr, y.iloc[tr])
        pred = model.predict_proba(X_va)[:, 1]
        fold_aucs.append(compute_auc(y.iloc[va], pred))
        encoded_names = encoded_feature_names(prep)
        encoded_count = len(encoded_names)
        coef_parts.append(pd.Series(np.abs(model.coef_[0]), index=encoded_names, name=f"fold_{fold}"))
        oof_parts.append(
            pd.DataFrame(
                {
                    "ID": X.index[va].astype(str),
                    "y_true": y.iloc[va].to_numpy(),
                    "fold": fold,
                    "model_name": model_name,
                    "model_type": "RidgeLogit",
                    "feature_set": feature_set,
                    "oof_pred": pred,
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    coef = pd.concat(coef_parts, axis=1)
    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": [original_feature_from_encoded(c) for c in coef.index],
            "encoded_feature": coef.index,
            "importance_mean": coef.mean(axis=1).to_numpy(),
            "importance_std": coef.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(original_feature_from_encoded(c), "encoded_or_unknown") for c in coef.index],
            "note": "absolute logistic coefficient",
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    note = "SGD log-loss quick fallback with fold-local OHE/imputer/scaler" if use_sgd else "Ridge logistic CV with fold-local OHE/imputer/scaler"
    row = cv_row(model_name, "RidgeLogit", feature_set, len(feature_cols), encoded_count, fold_aucs, [], oof, time.time() - start, note)
    return row, oof, importance


def cv_row(
    model_name: str,
    model_type: str,
    feature_set: str,
    feature_count: int,
    encoded_feature_count: float,
    fold_aucs: list[float],
    best_iterations: list[int],
    oof: pd.DataFrame,
    elapsed: float,
    note: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_type": model_type,
        "feature_set": feature_set,
        "feature_count": feature_count,
        "encoded_feature_count": encoded_feature_count,
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "training_time_sec": round(elapsed, 3),
        "delta_vs_catboost_baseline": np.nan,
        "note": note,
    }


def prediction_matrix(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.pivot_table(index="ID", columns="model_name", values="oof_pred")


def correlation_table(oof: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    rows = []
    for i, a in enumerate(mat.columns):
        for b in mat.columns[i + 1 :]:
            pair = mat[[a, b]].dropna()
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n": len(pair),
                    "pearson": pair[a].corr(pair[b], method="pearson") if len(pair) else np.nan,
                    "spearman": spearmanr(pair[a], pair[b]).correlation if len(pair) else np.nan,
                    "diversity_band": diversity_band(pair[a].corr(pair[b], method="pearson") if len(pair) else np.nan),
                }
            )
    return pd.DataFrame(rows)


def diversity_band(pearson: float) -> str:
    if pd.isna(pearson):
        return "unknown"
    if pearson >= 0.99:
        return "near duplicate"
    if pearson >= 0.96:
        return "weak diversity"
    if pearson >= 0.90:
        return "possible diversity"
    return "large diversity; verify performance"


def make_diagnostic_blends(oof: pd.DataFrame, cv_results: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    y = oof.drop_duplicates("ID").set_index("ID")["y_true"].loc[mat.index]
    if CB_NAME not in mat.columns:
        return pd.DataFrame()
    best_single_auc = cv_results["oof_auc"].max()
    cb_auc = float(cv_results.loc[cv_results["model_name"].eq(CB_NAME), "oof_auc"].iloc[0])
    non_cb = cv_results[~cv_results["model_name"].eq(CB_NAME)].sort_values("oof_auc", ascending=False)
    best_non_cb = non_cb.iloc[0]["model_name"] if not non_cb.empty else None
    xgb_rows = cv_results[cv_results["model_type"].eq("XGBoost")].sort_values("oof_auc", ascending=False)
    best_xgb = xgb_rows.iloc[0]["model_name"] if not xgb_rows.empty else None
    two_model_candidates = [
        "XGB_A_structured_numeric",
        "XGB_B_ohe_light_categorical",
        "ET_structured_numeric",
        "RidgeLogit_ohe_light",
    ]
    if best_non_cb:
        two_model_candidates.append(str(best_non_cb))
    rows: list[dict[str, Any]] = []
    for aux in dict.fromkeys(two_model_candidates):
        if aux not in mat.columns:
            continue
        for cb_w, aux_w in [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]:
            append_blend_row(rows, mat, y, [CB_NAME, aux], [cb_w, aux_w], best_single_auc, cb_auc, f"blend_{CB_NAME}_{aux}_{cb_w:.1f}_{aux_w:.1f}")
    three_specs = [
        [CB_NAME, best_xgb, LGBM_A_NAME],
        [CB_NAME, best_xgb, "ET_structured_numeric"],
        [CB_NAME, best_xgb, "RidgeLogit_ohe_light"],
    ]
    for names in three_specs:
        if any(name is None or name not in mat.columns for name in names):
            continue
        for weights in [(0.6, 0.25, 0.15), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
            append_blend_row(rows, mat, y, list(names), list(weights), best_single_auc, cb_auc, "blend_" + "_".join(names) + "_" + "_".join(str(w) for w in weights))
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False) if rows else pd.DataFrame()


def append_blend_row(
    rows: list[dict[str, Any]],
    mat: pd.DataFrame,
    y: pd.Series,
    names: list[str],
    weights: list[float],
    best_single_auc: float,
    cb_auc: float,
    blend_name: str,
) -> None:
    pair = mat[names].dropna()
    if pair.empty:
        return
    y_pair = y.loc[pair.index]
    pred = sum(w * pair[name] for w, name in zip(weights, names))
    pearsons = []
    spearmans = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pearsons.append(pair[a].corr(pair[b], method="pearson"))
            spearmans.append(spearmanr(pair[a], pair[b]).correlation)
    auc = compute_auc(y_pair, pred)
    rows.append(
        {
            "blend_name": blend_name,
            "base_models": ",".join(names),
            "weights": ",".join(str(w) for w in weights),
            "oof_auc": auc,
            "delta_vs_best_single": auc - best_single_auc,
            "delta_vs_catboost_baseline": auc - cb_auc,
            "mean_pairwise_pearson": float(np.nanmean(pearsons)),
            "mean_pairwise_spearman": float(np.nanmean(spearmans)),
            "note": "OOF-only diagnostic blend; not final ensemble tuning",
        }
    )


def branch_focus_diagnostics(subgroup: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
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
    for branch, (subgroup_name, value) in branch_map.items():
        sub = subgroup[subgroup["subgroup"].eq(subgroup_name) & subgroup["value"].astype(str).eq(value)]
        if sub.empty:
            continue
        cb = sub[sub["model_name"].eq(CB_NAME)]
        non_cb = sub[~sub["model_name"].eq(CB_NAME) & ~sub["model_name"].str.startswith("blend", na=False)]
        blends = sub[sub["model_name"].str.startswith("blend", na=False)]
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        best_non_cb = non_cb.sort_values("auc", ascending=False, na_position="last").head(1)
        cb_auc = cb["auc"].iloc[0] if len(cb) else np.nan
        aux_name = best_non_cb.iloc[0]["model_name"] if len(best_non_cb) else ""
        aux_auc = best_non_cb.iloc[0]["auc"] if len(best_non_cb) else np.nan
        blend_auc = blends["auc"].max() if len(blends) else np.nan
        rows.append(
            {
                "branch_name": branch,
                "n": int(best["n"]),
                "positive_rate": best["positive_rate"],
                "catboost_auc": cb_auc,
                "best_non_catboost_model": aux_name,
                "best_non_catboost_auc": aux_auc,
                "delta_best_non_catboost_vs_catboost": aux_auc - cb_auc if pd.notna(aux_auc) and pd.notna(cb_auc) else np.nan,
                "best_blend_auc": blend_auc,
                "delta_best_blend_vs_catboost": blend_auc - cb_auc if pd.notna(blend_auc) and pd.notna(cb_auc) else np.nan,
                "best_model_by_auc": best["model_name"],
                "interpretation": branch_interpretation(branch, cb_auc, aux_auc, blend_auc),
                "next_action": branch_next_action(branch, cb_auc, aux_auc, blend_auc),
            }
        )
    return pd.DataFrame(rows)


def branch_interpretation(branch: str, cb_auc: float, aux_auc: float, blend_auc: float) -> str:
    threshold = {"transfer_positive": 0.0005, "day5_transfer": 0.0005}.get(branch, 0.0007)
    if pd.notna(aux_auc) and pd.notna(cb_auc) and aux_auc >= cb_auc + threshold:
        return "non-CatBoost branch expert candidate"
    if pd.notna(blend_auc) and pd.notna(cb_auc) and blend_auc > cb_auc:
        return "blend/routing may add small value"
    return "CatBoost remains competitive"


def branch_next_action(branch: str, cb_auc: float, aux_auc: float, blend_auc: float) -> str:
    if branch in {"frozen", "donor_egg", "transfer_positive", "day5_transfer", "DI"} and pd.notna(aux_auc) and pd.notna(cb_auc) and aux_auc > cb_auc:
        return "add to soft-routing candidate pool"
    if pd.notna(blend_auc) and pd.notna(cb_auc) and blend_auc > cb_auc + 0.0002:
        return "controlled ensemble candidate"
    return "monitor"


def expert_pool_recommendation(branch: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in branch.iterrows():
        aux = r["best_non_catboost_model"]
        corr_val = corr_with_cb(corr, aux)
        delta = r["delta_best_non_catboost_vs_catboost"]
        strategy = "global_catboost_only"
        risk = "low"
        next_stage = "keep CatBoost as primary"
        if pd.notna(delta) and delta > 0.0007 and r["n"] >= 5000:
            strategy = "branch_soft_routing_candidate"
            risk = "medium"
            next_stage = "test controlled branch router"
        elif pd.notna(r["delta_best_blend_vs_catboost"]) and r["delta_best_blend_vs_catboost"] > 0.0002:
            strategy = "global_catboost_plus_aux_blend"
            risk = "medium" if r["n"] >= 5000 else "high"
            next_stage = "test low-weight controlled blend"
        elif r["n"] < 3000:
            strategy = "monitor_only"
            risk = "high"
            next_stage = "avoid specialization until evidence is stronger"
        rows.append(
            {
                "branch_name": r["branch_name"],
                "recommended_base_model": CB_NAME,
                "recommended_aux_model": aux,
                "recommended_strategy": strategy,
                "reason": r["interpretation"],
                "catboost_auc": r["catboost_auc"],
                "aux_model_auc": r["best_non_catboost_auc"],
                "delta_aux_vs_catboost": delta,
                "correlation_with_catboost": corr_val,
                "sample_size": r["n"],
                "risk_level": risk,
                "next_stage": next_stage,
            }
        )
    return pd.DataFrame(rows)


def corr_with_cb(corr: pd.DataFrame, model_name: str) -> float:
    if not model_name or corr.empty:
        return np.nan
    sub = corr[
        ((corr["model_a"].eq(CB_NAME)) & (corr["model_b"].eq(model_name)))
        | ((corr["model_b"].eq(CB_NAME)) & (corr["model_a"].eq(model_name)))
    ]
    return sub["pearson"].iloc[0] if len(sub) else np.nan


def optional_submission_sanity(note: str) -> pd.DataFrame:
    return pd.DataFrame(
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
                "note": note,
            }
        ]
    )


def optional_submission_allowed(cv_results: pd.DataFrame, blends: pd.DataFrame) -> tuple[bool, str]:
    cb_auc = float(cv_results.loc[cv_results["model_name"].eq(CB_NAME), "oof_auc"].iloc[0])
    non_cb = cv_results[~cv_results["model_name"].isin([CB_NAME, LGBM_A_NAME])]
    best_non_cb_auc = non_cb["oof_auc"].max() if not non_cb.empty else np.nan
    best_blend_auc = blends["oof_auc"].max() if not blends.empty else np.nan
    if pd.notna(best_non_cb_auc) and best_non_cb_auc >= cb_auc + 0.0002:
        return True, f"best non-CatBoost exceeded CatBoost by {best_non_cb_auc - cb_auc:.6f}"
    if pd.notna(best_blend_auc) and best_blend_auc >= PREVIOUS_BEST_BLEND_AUC + 0.0002:
        return True, f"best blend exceeded previous best blend by {best_blend_auc - PREVIOUS_BEST_BLEND_AUC:.6f}"
    if pd.notna(best_blend_auc) and best_blend_auc >= cb_auc + 0.0004:
        return True, f"best blend exceeded CatBoost by {best_blend_auc - cb_auc:.6f}"
    return False, "OOF criteria for optional submission were not met"


def build_report(tables: dict[str, pd.DataFrame], optional_note: str) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def table(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    cv = tables["model_zoo_cv_results"].sort_values("oof_auc", ascending=False)
    best = cv.iloc[0]
    non_cb = cv[~cv["model_name"].isin([CB_NAME])]
    best_non_cb = non_cb.sort_values("oof_auc", ascending=False).head(1)
    parts = [
        f"<style>{css}</style>",
        "<h1>Model Zoo v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best single model: {html.escape(str(best['model_name']))}, OOF AUC {best['oof_auc']:.6f}. Best non-CatBoost: {html.escape(str(best_non_cb.iloc[0]['model_name'])) if len(best_non_cb) else 'none'}.</p>",
        "<h2>Leakage and Prohibited Methods</h2>",
        "<p>No pseudo-labeling, self-training, target encoding, train+test concat, test EDA, test-wide post-processing, or automatic submission was used.</p>",
        "<h2>Why Model Zoo Was Needed</h2>",
        "<p>This stage searches for ranking diversity and branch specialists for later soft routing or controlled ensemble experiments.</p>",
        "<h2>Baseline CatBoost Recap</h2>",
        table("model_zoo_cv_results", 10),
        "<h2>Model Zoo CV Results</h2>",
        table("model_zoo_cv_results", 30),
        "<h2>OOF Correlation</h2>",
        table("model_zoo_oof_correlation", 50),
        "<h2>Diagnostic Blend Results</h2>",
        table("model_zoo_diagnostic_blend_results", 50),
        "<h2>Subgroup AUC</h2>",
        table("model_zoo_subgroup_auc", 80),
        "<h2>Branch Focus Diagnostics</h2>",
        table("model_zoo_branch_focus_diagnostics", 30),
        "<h2>Feature Importance</h2>",
        table("model_zoo_feature_importance", 80),
        "<h2>Expert Pool Recommendation</h2>",
        table("model_zoo_expert_pool_recommendation", 30),
        "<h2>Optional Submission Candidates</h2>",
        f"<p>{html.escape(optional_note)}</p>",
        table("optional_submission_sanity", 10),
        "<h2>Interpretation</h2>",
        "<p>Use non-CatBoost branch wins, lower CatBoost correlation, and OOF-only blend gains as evidence for the next controlled ensemble stage.</p>",
        "<h2>Next Step Recommendation</h2>",
        "<p>Promote only branches with sufficient sample size and positive OOF delta into a soft-routing experiment.</p>",
    ]
    (OUT_DIR / "model_zoo_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTIONAL_SUB_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []
    n_splits = 3 if args.quick else N_SPLITS
    xgb_a_config = dict(XGB_A_CONFIG)
    xgb_b_config = dict(XGB_A_CONFIG)
    xgb_c_config = dict(XGB_C_CONFIG)
    et_config = dict(ET_CONFIG)
    rf_config = dict(RF_CONFIG)
    logit_config = dict(LOGIT_CONFIG)
    if args.quick:
        xgb_a_config.update({"n_estimators": 120, "max_depth": 4, "early_stopping_rounds": 20, "n_jobs": 2})
        xgb_b_config.update({"n_estimators": 80, "max_depth": 4, "early_stopping_rounds": 20, "n_jobs": 2})
        xgb_c_config.update({"n_estimators": 120, "max_depth": 3, "early_stopping_rounds": 20, "n_jobs": 2})
        et_config.update({"n_estimators": 80, "n_jobs": 2})
        rf_config.update({"n_estimators": 80, "n_jobs": 2})
        logit_config.update({"max_iter": 120, "n_jobs": 2, "_use_sgd_quick": True})

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
                "xgb_a": xgb_a_config,
                "xgb_b": xgb_b_config,
                "xgb_c_optional": xgb_c_config,
                "extra_trees": et_config,
                "random_forest_optional": rf_config,
                "ridge_logit": logit_config,
                "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
                "quick_mode": args.quick,
                "include_optional": args.include_optional,
                "prohibited_methods": PROHIBITED_NOTE,
            },
        )

        result_frames: list[pd.DataFrame] = []
        oof_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []

        cb_oof, cb_result = load_catboost_oof(train)
        cb_result = cb_result.rename(columns={"categorical_feature_count": "encoded_feature_count"})
        cb_result["encoded_feature_count"] = cb_result.get("encoded_feature_count", np.nan)
        cb_result["delta_vs_catboost_baseline"] = 0.0
        result_frames.append(cb_result[["model_name", "model_type", "feature_set", "feature_count", "encoded_feature_count", "fold_auc_list", "mean_auc", "std_auc", "oof_auc", "best_iteration_list", "training_time_sec", "delta_vs_catboost_baseline", "note"]])
        cb_oof["model_type"] = "CatBoost"
        oof_frames.append(cb_oof[["ID", "y_true", "fold", "model_name", "model_type", "feature_set", "oof_pred"]])
        completed.append(f"{CB_NAME}_loaded")

        lgbm_oof, lgbm_result = load_lgbm_a_oof()
        if not lgbm_oof.empty:
            lgbm_result["delta_vs_catboost_baseline"] = np.nan
            result_frames.append(lgbm_result)
            lgbm_oof["model_type"] = "LightGBM"
            oof_frames.append(lgbm_oof[["ID", "y_true", "fold", "model_name", "model_type", "feature_set", "oof_pred"]])
            completed.append(f"{LGBM_A_NAME}_loaded")
        else:
            skipped.append(f"{LGBM_A_NAME}: previous OOF not found")

        model_runs = [
            ("xgb", "XGB_A_structured_numeric", "structured_numeric", xgb_a_config),
            ("xgb", "XGB_B_ohe_light_categorical", "ohe_light_categorical", xgb_b_config),
            ("tree", "ET_structured_numeric", "structured_numeric", et_config),
            ("logit", "RidgeLogit_ohe_light", "ohe_light_categorical", logit_config),
        ]
        if args.include_optional:
            model_runs.extend(
                [
                    ("xgb", "XGB_C_shallow_regularized", "structured_numeric", xgb_c_config),
                    ("tree_rf", "RF_structured_numeric_optional", "structured_numeric", rf_config),
                ]
            )
        else:
            skipped.extend(["XGB_C_shallow_regularized: optional not requested", "RF_structured_numeric_optional: optional not requested"])

        for kind, model_name, feature_set, config in model_runs:
            try:
                print(f"[model_zoo_v1] starting {model_name}", flush=True)
                if kind == "xgb":
                    row, oof, imp = run_xgb_cv(X, y, feature_sets[feature_set], model_name, feature_set, config, n_splits, lookup)
                elif kind == "tree":
                    row, oof, imp = run_tree_cv(X, y, feature_sets[feature_set], model_name, "ExtraTrees", feature_set, config, n_splits, lookup)
                elif kind == "tree_rf":
                    row, oof, imp = run_tree_cv(X, y, feature_sets[feature_set], model_name, "RandomForest", feature_set, config, n_splits, lookup)
                else:
                    row, oof, imp = run_logit_cv(X, y, feature_sets[feature_set], config, n_splits, lookup)
                result_frames.append(pd.DataFrame([row]))
                oof_frames.append(oof)
                importance_frames.append(imp)
                completed.append(model_name)
                print(f"[model_zoo_v1] completed {model_name}: oof_auc={row['oof_auc']:.6f}", flush=True)
            except Exception as exc:
                skipped.append(f"{model_name}: {type(exc).__name__}: {exc}")
                print(f"[model_zoo_v1] skipped {model_name}: {type(exc).__name__}: {exc}", flush=True)

        cv_results = pd.concat(result_frames, ignore_index=True)
        cb_auc = float(cv_results.loc[cv_results["model_name"].eq(CB_NAME), "oof_auc"].iloc[0])
        cv_results["delta_vs_catboost_baseline"] = cv_results["oof_auc"] - cb_auc
        cv_results = cv_results.sort_values("oof_auc", ascending=False)
        oof_all = pd.concat(oof_frames, ignore_index=True)
        importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        corr = correlation_table(oof_all)
        blends = make_diagnostic_blends(oof_all, cv_results, corr)
        subgroup = subgroup_auc_by_model(X, train, oof_all.drop(columns=["model_type"], errors="ignore"), blends)
        branch = branch_focus_diagnostics(subgroup, corr)
        expert_pool = expert_pool_recommendation(branch, corr)
        allowed, optional_note = optional_submission_allowed(cv_results, blends)
        if args.make_optional_submissions and allowed:
            skipped.append("optional_submission_generation: criteria met but diagnostics-only implementation did not fit test candidates")
            optional_note += "; candidate creation deferred to controlled ensemble stage"
        elif args.make_optional_submissions:
            skipped.append(f"optional_submission_generation: {optional_note}")
        else:
            skipped.append("optional_submission_generation: disabled by default")
            optional_note = "Optional submissions disabled by default; " + optional_note
        optional_sanity = optional_submission_sanity(optional_note)

        tables = {
            "model_zoo_cv_results": cv_results,
            "model_zoo_oof_predictions": oof_all,
            "model_zoo_oof_correlation": corr,
            "model_zoo_diagnostic_blend_results": blends,
            "model_zoo_subgroup_auc": subgroup,
            "model_zoo_branch_focus_diagnostics": branch,
            "model_zoo_feature_importance": importance,
            "model_zoo_expert_pool_recommendation": expert_pool,
            "optional_submission_sanity": optional_sanity,
        }
        for key, df in tables.items():
            if key == "optional_submission_sanity":
                save_table(OUT_DIR / "optional_submission_sanity.csv", df)
            else:
                save_table(OUT_DIR / f"{key}.csv", df)
        build_report(tables, optional_note)

        warnings_list.extend(str(w.message) for w in caught)
        import sklearn
        import xgboost

        best_single = cv_results.iloc[0]
        non_cb = cv_results[~cv_results["model_name"].eq(CB_NAME)].sort_values("oof_auc", ascending=False)
        best_non_cb = non_cb.iloc[0] if not non_cb.empty else pd.Series({"model_name": "none", "oof_auc": np.nan, "delta_vs_catboost_baseline": np.nan})
        best_blend = blends.iloc[0] if not blends.empty else pd.Series({"blend_name": "none", "oof_auc": np.nan, "base_models": "", "weights": ""})
        branch_success = branch[branch["next_action"].str.contains("soft-routing|controlled ensemble", case=False, na=False)].to_dict(orient="records")
        expert_summary = expert_pool[expert_pool["recommended_strategy"].ne("global_catboost_only")].to_dict(orient="records")
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"sklearn_version: {sklearn.__version__}",
            f"xgboost_version: {xgboost.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"test_path: {TEST_PATH}",
            f"target_column: {target}",
            f"cv_setting: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
            f"baseline_catboost_oof_auc: {cb_auc:.6f}",
            f"model_settings: xgb_a={xgb_a_config}; xgb_b={xgb_b_config}; et={et_config}; logit={logit_config}; optional_xgb_c={xgb_c_config}; optional_rf={rf_config}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"best_single_model: {best_single['model_name']}",
            f"best_single_oof_auc: {best_single['oof_auc']:.6f}",
            f"best_non_catboost_model: {best_non_cb['model_name']}",
            f"best_non_catboost_oof_auc: {best_non_cb['oof_auc']}",
            f"best_diagnostic_blend: {best_blend['blend_name']}",
            f"best_diagnostic_blend_oof_auc: {best_blend['oof_auc']}",
            f"branch_success_summary: {branch_success}",
            f"expert_pool_summary: {expert_summary}",
            "optional_candidate_paths: none",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "model_zoo_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
