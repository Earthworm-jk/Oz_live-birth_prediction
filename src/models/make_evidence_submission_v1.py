from __future__ import annotations

import json
import platform
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.model_selection import StratifiedKFold

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.controlled_full_cv_v1 import all_cols_from_families
from src.models.evidence_guided_feature_audit_v1 import (
    PROHIBITED_NOTE,
    V2_CONFIGS,
    add_evidence_features,
    build_feature_sets,
)
from src.models.model_branch_diagnostics_v1 import (
    EARLY_STOPPING_ROUNDS,
    LGBM_CONFIG,
    categorical_columns,
    make_feature_sets as make_lgbm_feature_sets,
    make_lgbm,
    prepare_lgbm_fold,
    prepare_lgbm_full,
)
from src.models.model_utils import get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    RAW_DIR / "sample.csv",
    RAW_DIR / "submission_sample.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
OUT_DIR = PROJECT_ROOT / "outputs" / "evidence_submission_v1"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
RANDOM_SEED = 42
N_SPLITS = 5
TARGET_HINT = "임신 성공 여부"

OLD_CB_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "candidate_catboost_long_all_features.csv",
    PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "submission_catboost_long_all_features.csv",
    PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "optional_submissions" / "candidate_catboost_long_all_features.csv",
    PROJECT_ROOT / "outputs" / "controlled_full_cv_v1" / "candidate_catboost_long_all_features.csv",
    PROJECT_ROOT / "outputs" / "candidate_catboost_long_all_features.csv",
    PROJECT_ROOT / "submissions" / "candidate_catboost_long_all_features.csv",
]
LGBM_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "test_pred_lgbm_a_all_features.csv",
    PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "candidate_lgbm_a_all_features.csv",
    PROJECT_ROOT / "outputs" / "model_zoo_v1" / "test_pred_lgbm_a_all_features.csv",
    PROJECT_ROOT / "outputs" / "candidate_lgbm_a_all_features.csv",
]

CANDIDATE_WEIGHTS = {
    "candidate_cbv2_depth7_all_evidence": {"cbv2_depth7": 1.0},
    "candidate_cbv2_depth6_all_evidence": {"cbv2_depth6": 1.0},
    "candidate_cbv2_lgbm_070_030": {"cbv2_depth7": 0.7, "lgbm_a": 0.3},
    "candidate_cbv2_oldcb_lgbm_060_020_020": {"cbv2_depth7": 0.6, "old_cb": 0.2, "lgbm_a": 0.2},
}


def find_sample() -> Path:
    for path in SAMPLE_SUBMISSION_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(f"sample submission not found under: {SAMPLE_SUBMISSION_CANDIDATES}")


def prediction_column(sample: pd.DataFrame) -> str:
    cols = [c for c in sample.columns if c != ID_COLUMN]
    if len(cols) != 1:
        raise ValueError(f"Expected one prediction column in sample submission, got: {cols}")
    return cols[0]


def probability_check(name: str, pred: np.ndarray) -> None:
    if np.isnan(pred).any():
        raise ValueError(f"{name}: prediction contains NaN")
    if pred.min() < 0 or pred.max() > 1:
        raise ValueError(f"{name}: prediction outside [0, 1], min={pred.min()}, max={pred.max()}")


def save_prediction(path: Path, ids: pd.Series, pred: np.ndarray, pred_col: str = "prediction") -> None:
    probability_check(path.stem, pred)
    save_table(path, pd.DataFrame({ID_COLUMN: ids.astype(str).to_numpy(), pred_col: pred}))


def make_submission(path: Path, sample: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    probability_check(path.stem, pred)
    sub = sample.copy()
    pred_col = prediction_column(sub)
    sub[pred_col] = pred
    save_table(path, sub)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    save_table(SUBMISSION_DIR / path.name, sub)
    return sub


def feature_columns_all_evidence(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X_base = make_art_features(train_raw)
    X_base.index = train_raw[ID_COLUMN].astype(str)
    X, feature_list, skipped = add_evidence_features(X_base, train_raw)
    if not skipped.empty:
        raise ValueError(f"Evidence feature creation skipped train features: {skipped.to_dict(orient='records')}")
    base_families = save_feature_families(OUT_DIR / "feature_families_base.json", X_base.columns.tolist())
    base_cols = all_cols_from_families(base_families, X_base)
    feature_sets = build_feature_sets(base_cols, feature_list)
    return X, feature_sets["all_evidence_features"]


def transform_test_evidence(test_raw: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    X_base = make_art_features(test_raw)
    X_base.index = test_raw[ID_COLUMN].astype(str)
    X, _, skipped = add_evidence_features(X_base, test_raw)
    if not skipped.empty:
        raise ValueError(f"Evidence feature creation skipped test features: {skipped.to_dict(orient='records')}")
    missing = [c for c in train_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in set(train_cols) and c not in X_base.columns]
    if missing:
        raise ValueError(f"Test feature columns missing: {missing[:20]}")
    if extra:
        # Extra evidence columns would mean train/test feature creation diverged.
        raise ValueError(f"Unexpected test-only evidence columns: {extra[:20]}")
    return X[train_cols]


def predict_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    model_name: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    from catboost import CatBoostClassifier

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    preds = []
    best_iterations = []
    fold_aucs = []
    for fold, (tr, va) in enumerate(cv.split(X[feature_cols], y), start=1):
        start = time.time()
        X_tr, X_va = X.iloc[tr][feature_cols].copy(), X.iloc[va][feature_cols].copy()
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        X_pred = X_test[feature_cols].copy()
        X_tr, cat_cols = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        X_pred, _ = prepare_catboost_frame(X_pred)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model = CatBoostClassifier(**config)
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        va_pred = model.predict_proba(X_va)[:, 1]
        test_pred = model.predict_proba(X_pred)[:, 1]
        from sklearn.metrics import roc_auc_score

        fold_auc = float(roc_auc_score(y_va, va_pred))
        best_iter = int(model.get_best_iteration() or config["iterations"])
        preds.append(test_pred)
        fold_aucs.append(fold_auc)
        best_iterations.append(best_iter)
        print(f"[evidence_submission] {model_name} fold {fold} done auc={fold_auc:.6f} best_iter={best_iter} sec={time.time()-start:.1f}", flush=True)
    pred = np.mean(np.vstack(preds), axis=0)
    probability_check(model_name, pred)
    return pred, best_iterations, fold_aucs


def load_existing_prediction(paths: list[Path], sample: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    pred_col = prediction_column(sample)
    sample_ids = sample[ID_COLUMN].astype(str)
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if ID_COLUMN not in df.columns:
            continue
        candidate_cols = [c for c in df.columns if c != ID_COLUMN]
        if not candidate_cols:
            continue
        col = pred_col if pred_col in candidate_cols else candidate_cols[0]
        if len(df) != len(sample):
            continue
        if not df[ID_COLUMN].astype(str).equals(sample_ids):
            continue
        pred = pd.to_numeric(df[col], errors="coerce").to_numpy()
        probability_check(path.name, pred)
        return pred, str(path)
    return None, ""


def predict_lgbm_cv(train_raw: pd.DataFrame, test_raw: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[int]]:
    from lightgbm import early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score

    X = make_art_features(train_raw)
    X.index = train_raw[ID_COLUMN].astype(str)
    X_test = make_art_features(test_raw)
    X_test.index = test_raw[ID_COLUMN].astype(str)
    families = save_feature_families(OUT_DIR / "feature_families_lgbm.json", X.columns.tolist())
    feature_sets = make_lgbm_feature_sets(X, families)
    cols = [c for c in feature_sets["all_features_for_lgbm"] if c in X.columns]
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise ValueError(f"LGBM test features missing: {missing[:20]}")

    config = dict(LGBM_CONFIG)
    preds = []
    best_iterations = []
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cat_cols = categorical_columns(X[cols])
    for fold, (tr, va) in enumerate(cv.split(X[cols], y), start=1):
        X_tr_raw = X.iloc[tr][cols]
        X_va_raw = X.iloc[va][cols]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        X_tr, X_va, fit_cat_cols = prepare_lgbm_fold(X_tr_raw, X_va_raw, cat_cols)
        X_tr_full, X_pred, fit_cat_cols_full = prepare_lgbm_full(X_tr_raw, X_test[cols], cat_cols)
        # Use validation-prepared train data for early stopping; then predict with test categories aligned to the same fold train.
        model = make_lgbm(config)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=fit_cat_cols,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
        va_pred = model.predict_proba(X_va)[:, 1]
        test_pred = model.predict_proba(X_pred)[:, 1]
        preds.append(test_pred)
        best_iterations.append(int(model.best_iteration_ or config["n_estimators"]))
        print(f"[evidence_submission] LGBM_A fold {fold} done auc={roc_auc_score(y_va, va_pred):.6f} best_iter={best_iterations[-1]}", flush=True)
    pred = np.mean(np.vstack(preds), axis=0)
    probability_check("LGBM_A_all_features", pred)
    return pred, best_iterations


def sanity_row(candidate: str, path: Path, sub: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    pred_col = prediction_column(sample)
    pred = pd.to_numeric(sub[pred_col], errors="coerce")
    return {
        "candidate": candidate,
        "path": str(path),
        "row_count": len(sub),
        "row_count_equals_sample": len(sub) == len(sample),
        "id_order_matches_sample": sub[ID_COLUMN].astype(str).equals(sample[ID_COLUMN].astype(str)),
        "prediction_has_no_nan": pred.notna().all(),
        "prediction_min": pred.min(),
        "prediction_max": pred.max(),
        "prediction_mean": pred.mean(),
        "prediction_std": pred.std(),
        "prediction_unique_count": int(pred.nunique(dropna=True)),
        "duplicated_id_count": int(sub[ID_COLUMN].duplicated().sum()),
        "is_probability_range": bool(pred.min() >= 0 and pred.max() <= 1),
        "note": "submission candidate; not automatically submitted",
    }


def prediction_summary(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in preds.items():
        s = pd.Series(pred)
        rows.append(
            {
                "model_or_candidate": name,
                "prediction_mean": s.mean(),
                "prediction_std": s.std(),
                "prediction_min": s.min(),
                "prediction_p01": s.quantile(0.01),
                "prediction_p05": s.quantile(0.05),
                "prediction_p25": s.quantile(0.25),
                "prediction_p50": s.quantile(0.50),
                "prediction_p75": s.quantile(0.75),
                "prediction_p95": s.quantile(0.95),
                "prediction_p99": s.quantile(0.99),
                "prediction_max": s.max(),
                "correlation_with_cbv2_depth7": s.corr(pd.Series(preds["cbv2_depth7"])) if "cbv2_depth7" in preds else np.nan,
                "correlation_with_old_cb": s.corr(pd.Series(preds["old_cb"])) if "old_cb" in preds else np.nan,
                "correlation_with_lgbm": s.corr(pd.Series(preds["lgbm_a"])) if "lgbm_a" in preds else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        train_raw = pd.read_csv(TRAIN_PATH)
        test_raw = pd.read_csv(TEST_PATH)
        sample_path = find_sample()
        sample = pd.read_csv(sample_path)
        pred_col = prediction_column(sample)
        target = get_target_column(train_raw)
        y = train_raw[target].astype(int)
        if not test_raw[ID_COLUMN].astype(str).equals(sample[ID_COLUMN].astype(str)):
            raise ValueError("test ID order does not match sample_submission ID order")

        X_train, feature_cols = feature_columns_all_evidence(train_raw)
        X_test = transform_test_evidence(test_raw, feature_cols)
        if list(X_train[feature_cols].columns) != list(X_test.columns):
            raise ValueError("train/test feature order mismatch")

        configs = {
            "cbv2_depth7": dict(V2_CONFIGS["CB_v2_depth7_lr045_l2_6"]),
            "cbv2_depth6": dict(V2_CONFIGS["CB_v2_depth6_lr035_l2_5"]),
        }
        pred_depth7, depth7_iters, depth7_aucs = predict_catboost_cv(X_train, y, X_test, feature_cols, configs["cbv2_depth7"], "CB_v2_depth7_lr045_l2_6")
        save_prediction(OUT_DIR / "test_pred_cbv2_depth7_all_evidence.csv", sample[ID_COLUMN], pred_depth7)

        pred_depth6, depth6_iters, depth6_aucs = predict_catboost_cv(X_train, y, X_test, feature_cols, configs["cbv2_depth6"], "CB_v2_depth6_lr035_l2_5")
        save_prediction(OUT_DIR / "test_pred_cbv2_depth6_all_evidence.csv", sample[ID_COLUMN], pred_depth6)

        old_cb_pred, old_cb_source = load_existing_prediction(OLD_CB_CANDIDATES, sample)
        old_regenerated = False
        if old_cb_pred is None:
            raise FileNotFoundError("Old CatBoost prediction was not found; regeneration is intentionally not implemented in this submission script.")
        save_prediction(OUT_DIR / "test_pred_old_catboost_long.csv", sample[ID_COLUMN], old_cb_pred)

        lgbm_pred, lgbm_source = load_existing_prediction(LGBM_CANDIDATES, sample)
        lgbm_regenerated = False
        lgbm_iters: list[int] = []
        if lgbm_pred is None:
            lgbm_pred, lgbm_iters = predict_lgbm_cv(train_raw, test_raw, y)
            lgbm_source = "regenerated_cv_ensemble"
            lgbm_regenerated = True
        save_prediction(OUT_DIR / "test_pred_lgbm_a_all_features.csv", sample[ID_COLUMN], lgbm_pred)

        base_preds = {
            "cbv2_depth7": pred_depth7,
            "cbv2_depth6": pred_depth6,
            "old_cb": old_cb_pred,
            "lgbm_a": lgbm_pred,
        }
        candidate_preds = {
            "candidate_cbv2_depth7_all_evidence": pred_depth7,
            "candidate_cbv2_depth6_all_evidence": pred_depth6,
            "candidate_cbv2_lgbm_070_030": 0.7 * pred_depth7 + 0.3 * lgbm_pred,
            "candidate_cbv2_oldcb_lgbm_060_020_020": 0.6 * pred_depth7 + 0.2 * old_cb_pred + 0.2 * lgbm_pred,
        }
        sanity_rows = []
        candidate_paths = []
        for name, pred in candidate_preds.items():
            path = OUT_DIR / f"{name}.csv"
            sub = make_submission(path, sample, pred)
            sanity_rows.append(sanity_row(name, path, sub, sample))
            candidate_paths.append(path)

        all_preds = {**base_preds, **candidate_preds}
        summary = prediction_summary(all_preds)
        sanity = pd.DataFrame(sanity_rows)
        save_table(OUT_DIR / "evidence_submission_prediction_summary.csv", summary)
        save_table(OUT_DIR / "evidence_submission_sanity.csv", sanity)

        zip_path = OUT_DIR / "evidence_submission_candidates.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in candidate_paths:
                zf.write(path, arcname=path.name)

        config = {
            "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
            "target": target,
            "id_column": ID_COLUMN,
            "feature_set": "all_evidence_features",
            "feature_count": len(feature_cols),
            "categorical_count": len(categorical_columns(X_train[feature_cols])),
            "cb_v2_depth7_config": configs["cbv2_depth7"],
            "cb_v2_depth6_config": configs["cbv2_depth6"],
            "candidate_weights": CANDIDATE_WEIGHTS,
            "old_cb_prediction_source": old_cb_source,
            "lgbm_prediction_source": lgbm_source,
            "prohibited_methods": PROHIBITED_NOTE,
        }
        save_json(OUT_DIR / "evidence_submission_config.json", config)

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import lightgbm
        import sklearn

        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"sklearn_version: {sklearn.__version__}",
            f"catboost_version: {catboost.__version__}",
            f"lightgbm_version: {lightgbm.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"test_path: {TEST_PATH}",
            f"sample_path: {sample_path}",
            f"target_column: {target}",
            f"id_column: {ID_COLUMN}",
            "feature_set: all_evidence_features",
            f"feature_count: {len(feature_cols)}",
            f"categorical_count: {len(categorical_columns(X_train[feature_cols]))}",
            f"cv_setting: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
            "models_trained: CB_v2_depth7_lr045_l2_6,CB_v2_depth6_lr035_l2_5,LGBM_A_all_features" if lgbm_regenerated else "models_trained: CB_v2_depth7_lr045_l2_6,CB_v2_depth6_lr035_l2_5",
            f"existing_prediction_files_reused: old_cb={old_cb_source}; lgbm={'' if lgbm_regenerated else lgbm_source}",
            f"regenerated_prediction: old_cb={old_regenerated}; lgbm={lgbm_regenerated}",
            f"fold_best_iteration_list_cbv2_depth7: {depth7_iters}",
            f"fold_auc_list_cbv2_depth7: {depth7_aucs}",
            f"fold_best_iteration_list_cbv2_depth6: {depth6_iters}",
            f"fold_auc_list_cbv2_depth6: {depth6_aucs}",
            f"fold_best_iteration_list_lgbm: {lgbm_iters}",
            f"oldCB_prediction_source: {old_cb_source}",
            f"LGBM_prediction_source: {lgbm_source}",
            f"generated_candidate_paths: {[str(p) for p in candidate_paths]}",
            f"submission_folder_copies: {[str(SUBMISSION_DIR / p.name) for p in candidate_paths]}",
            f"sanity_check_summary: {sanity.to_dict(orient='records')}",
            f"zip_path: {zip_path}",
            f"prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "evidence_submission_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
