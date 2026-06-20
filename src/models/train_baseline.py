from __future__ import annotations

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
from sklearn.model_selection import StratifiedKFold

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.model_utils import (
    MISSING_CATEGORY,
    compute_auc,
    get_categorical_columns,
    compute_subgroup_auc,
    get_target_column,
    prepare_catboost_frame,
    save_json,
    save_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]

RANDOM_SEED = 42
LEAKAGE_NOTE = (
    "All feature engineering is row-wise or fitted on training folds only. "
    "Test data is used only for schema-compatible transformation and prediction, "
    "not for EDA or fitting preprocessing statistics."
)
SUBMISSION_NOTE = "This file is a submission candidate only. It was not submitted automatically."


MODEL_CONFIG: dict[str, Any] = {
    "model_type": "CatBoostClassifier",
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 180,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_SEED,
    "early_stopping_rounds": 30,
    "verbose": False,
    "allow_writing_files": False,
    "cv_n_splits": 5,
    "ablation_experiments_run": ["raw_catboost", "plus_treatment_reason", "plus_age_transfer", "plus_history_cause"],
    "note": "Fast baseline: 5-fold CV kept, CatBoost iterations reduced so the full artifact set finishes in this workspace.",
}


EXPERIMENTS = [
    ("raw_catboost", ["raw_base"]),
    ("raw_plus_branch", ["raw_base", "branch_flags"]),
    ("plus_treatment_reason", ["raw_base", "branch_flags", "treatment_tokens", "embryo_reason_features"]),
    (
        "plus_age_transfer",
        [
            "raw_base",
            "branch_flags",
            "treatment_tokens",
            "embryo_reason_features",
            "age_source_interactions",
            "transfer_features",
        ],
    ),
    (
        "plus_funnel_counts",
        [
            "raw_base",
            "branch_flags",
            "treatment_tokens",
            "embryo_reason_features",
            "age_source_interactions",
            "transfer_features",
            "funnel_count_bin_features",
        ],
    ),
    (
        "plus_funnel_ratios",
        [
            "raw_base",
            "branch_flags",
            "treatment_tokens",
            "embryo_reason_features",
            "age_source_interactions",
            "transfer_features",
            "funnel_count_bin_features",
            "funnel_ratio_features",
        ],
    ),
    (
        "plus_history_cause",
        [
            "raw_base",
            "branch_flags",
            "treatment_tokens",
            "embryo_reason_features",
            "age_source_interactions",
            "transfer_features",
            "funnel_count_bin_features",
            "funnel_ratio_features",
            "history_features",
            "cause_features",
        ],
    ),
]


def make_model():
    from catboost import CatBoostClassifier

    params = {
        "loss_function": MODEL_CONFIG["loss_function"],
        "eval_metric": MODEL_CONFIG["eval_metric"],
        "iterations": MODEL_CONFIG["iterations"],
        "learning_rate": MODEL_CONFIG["learning_rate"],
        "depth": MODEL_CONFIG["depth"],
        "l2_leaf_reg": MODEL_CONFIG["l2_leaf_reg"],
        "random_seed": MODEL_CONFIG["random_seed"],
        "early_stopping_rounds": MODEL_CONFIG["early_stopping_rounds"],
        "verbose": MODEL_CONFIG["verbose"],
        "allow_writing_files": MODEL_CONFIG["allow_writing_files"],
    }
    return CatBoostClassifier(**params)


def feature_columns_for_experiment(families: dict[str, list[str]], selected_families: list[str]) -> list[str]:
    cols: list[str] = []
    for family in selected_families:
        cols.extend(families.get(family, []))
    return list(dict.fromkeys(cols))


def run_cv_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    experiment_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any], pd.DataFrame]:
    skf = StratifiedKFold(n_splits=MODEL_CONFIG["cv_n_splits"], shuffle=True, random_state=RANDOM_SEED)
    rows = []
    oof_parts = []
    models = []
    importances = []
    start = time.time()

    X_exp = X[feature_columns].copy()
    cat_cols = get_categorical_columns(X_exp)
    num_cols = [c for c in X_exp.columns if c not in cat_cols]

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_exp, y), start=1):
        fold_start = time.time()
        X_train, X_valid = X_exp.iloc[train_idx].copy(), X_exp.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_train, cat_cols_train = prepare_catboost_frame(X_train)
        X_valid, _ = prepare_catboost_frame(X_valid)
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols_train]

        model = make_model()
        model.fit(
            X_train,
            y_train,
            cat_features=cat_indices,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )
        pred = model.predict_proba(X_valid)[:, 1]
        auc = compute_auc(y_valid, pred)
        rows.append(
            {
                "experiment": experiment_name,
                "fold": fold,
                "auc": auc,
                "best_iteration": int(model.get_best_iteration() or MODEL_CONFIG["iterations"]),
                "feature_count": len(feature_columns),
                "categorical_feature_count": len(cat_cols),
                "numeric_feature_count": len(num_cols),
                "training_time_sec": round(time.time() - fold_start, 3),
                "notes": MODEL_CONFIG["note"],
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_index": valid_idx,
                    "y_true": y_valid.to_numpy(),
                    "oof_pred": pred,
                    "fold": fold,
                    "experiment": experiment_name,
                }
            )
        )
        models.append(model)
        importances.append(pd.Series(model.get_feature_importance(), index=feature_columns, name=f"fold_{fold}"))

    cv = pd.DataFrame(rows)
    cv["experiment_training_time_sec"] = round(time.time() - start, 3)
    oof = pd.concat(oof_parts, ignore_index=True)
    importance = pd.concat(importances, axis=1) if importances else pd.DataFrame(index=feature_columns)
    return cv, oof, models, importance


def run_ablation_suite(X: pd.DataFrame, y: pd.Series, families: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Any]], dict[str, pd.DataFrame]]:
    cv_frames = []
    oof_frames = []
    model_map: dict[str, list[Any]] = {}
    importance_map: dict[str, pd.DataFrame] = {}
    selected = set(MODEL_CONFIG["ablation_experiments_run"])

    for experiment_name, family_names in EXPERIMENTS:
        if experiment_name not in selected:
            continue
        cols = feature_columns_for_experiment(families, family_names)
        cv, oof, models, importance = run_cv_experiment(X, y, cols, experiment_name)
        cv_frames.append(cv)
        oof_frames.append(oof)
        model_map[experiment_name] = models
        importance_map[experiment_name] = importance

    cv_results = pd.concat(cv_frames, ignore_index=True)
    oof_all = pd.concat(oof_frames, ignore_index=True)
    return cv_results, oof_all, model_map, importance_map


def summarize_ablation(cv_results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        cv_results.groupby("experiment")
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            fold_auc_list=("auc", lambda s: ",".join(f"{v:.6f}" for v in s)),
            feature_count=("feature_count", "first"),
            categorical_feature_count=("categorical_feature_count", "first"),
            numeric_feature_count=("numeric_feature_count", "first"),
            training_time_sec=("training_time_sec", "sum"),
            notes=("notes", "first"),
        )
        .reset_index()
        .sort_values("mean_auc", ascending=False)
    )
    return grouped


def best_experiment_from(ablation: pd.DataFrame) -> str:
    return str(ablation.sort_values("mean_auc", ascending=False).iloc[0]["experiment"])


def attach_ids_to_oof(oof: pd.DataFrame, train: pd.DataFrame, experiment: str) -> pd.DataFrame:
    best = oof[oof["experiment"].eq(experiment)].copy()
    best["ID"] = train.iloc[best["row_index"]][ID_COLUMN].to_numpy() if ID_COLUMN in train.columns else best["row_index"].astype(str)
    return best[["ID", "y_true", "oof_pred", "fold"]].sort_values("ID")


def make_subgroup_auc_results(train_features: pd.DataFrame, train_raw: pd.DataFrame, oof_best: pd.DataFrame) -> pd.DataFrame:
    frame = train_features.copy()
    frame["ID"] = train_raw[ID_COLUMN].to_numpy() if ID_COLUMN in train_raw.columns else np.arange(len(frame)).astype(str)
    merged = frame.merge(oof_best, on="ID", how="inner")
    rows = []
    rows.append(compute_subgroup_auc(merged, "y_true", "oof_pred", "overall", pd.Series(True, index=merged.index), "all"))

    subgroup_specs = [
        ("treatment_type", "IVF", merged.get("시술 유형", pd.Series(index=merged.index)).eq("IVF")),
        ("treatment_type", "DI", merged.get("시술 유형", pd.Series(index=merged.index)).eq("DI")),
        ("age_group", "all", merged.get("age_group_raw", pd.Series(index=merged.index)).notna()),
        ("egg_source", "본인 제공", merged.get("난자 출처", pd.Series(index=merged.index)).eq("본인 제공")),
        ("egg_source", "기증 제공", merged.get("난자 출처", pd.Series(index=merged.index)).eq("기증 제공")),
        ("is_donor_egg", "1", merged.get("is_donor_egg", pd.Series(0, index=merged.index)).eq(1)),
        ("is_donor_egg", "0", merged.get("is_donor_egg", pd.Series(0, index=merged.index)).eq(0)),
        ("embryo_transferred_flag", "1", merged.get("embryo_transferred_flag", pd.Series(0, index=merged.index)).eq(1)),
        ("no_embryo_transfer_flag", "1", merged.get("no_embryo_transfer_flag", pd.Series(0, index=merged.index)).eq(1)),
        ("is_fresh_embryo", "1", merged.get("is_fresh_embryo", pd.Series(0, index=merged.index)).eq(1)),
        ("is_frozen_embryo", "1", merged.get("is_frozen_embryo", pd.Series(0, index=merged.index)).eq(1)),
        ("reason_current_treatment", "1", merged.get("reason_current_treatment", pd.Series(0, index=merged.index)).eq(1)),
        ("storage_only_flag", "1", merged.get("storage_only_flag", pd.Series(0, index=merged.index)).eq(1)),
        ("donation_only_flag", "1", merged.get("donation_only_flag", pd.Series(0, index=merged.index)).eq(1)),
        ("transfer_day_5", "1", merged.get("transfer_day_5", pd.Series(0, index=merged.index)).eq(1)),
        ("transfer_day_missing", "1", merged.get("transfer_day_missing", pd.Series(0, index=merged.index)).eq(1)),
    ]
    for name, value, mask in subgroup_specs:
        rows.append(compute_subgroup_auc(merged, "y_true", "oof_pred", name, mask, value))

    for age_value in sorted(merged.get("age_group_raw", pd.Series(dtype=object)).dropna().unique()):
        rows.append(compute_subgroup_auc(merged, "y_true", "oof_pred", "age_group", merged["age_group_raw"].eq(age_value), str(age_value)))
    return pd.DataFrame(rows)


def build_feature_importance(
    importance: pd.DataFrame,
    families: dict[str, list[str]],
) -> pd.DataFrame:
    feature_family = {}
    for family, cols in families.items():
        for col in cols:
            feature_family[col] = family
    out = pd.DataFrame(
        {
            "feature": importance.index,
            "importance_mean": importance.mean(axis=1).to_numpy(),
            "importance_std": importance.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [feature_family.get(c, "unknown") for c in importance.index],
        }
    )
    return out.sort_values("importance_mean", ascending=False)


def train_final_model(X: pd.DataFrame, y: pd.Series, feature_columns: list[str]):
    X_final, cat_cols = prepare_catboost_frame(X[feature_columns].copy())
    cat_indices = [X_final.columns.get_loc(c) for c in cat_cols]
    model = make_model()
    model.fit(X_final, y, cat_features=cat_indices)
    return model, cat_cols


def create_submission_candidate(model: Any, test_features: pd.DataFrame, feature_columns: list[str], test_raw: pd.DataFrame) -> Path:
    X_test, _ = prepare_catboost_frame(test_features[feature_columns].copy())
    preds = model.predict_proba(X_test)[:, 1]
    sample_path = next((p for p in SAMPLE_SUBMISSION_CANDIDATES if p.exists()), None)
    if sample_path is not None:
        submission = pd.read_csv(sample_path)
        pred_col = [c for c in submission.columns if c != ID_COLUMN][0]
        submission[pred_col] = preds
    else:
        submission = pd.DataFrame({ID_COLUMN: test_raw[ID_COLUMN], "probability": preds})
    path = SUBMISSION_DIR / "candidate_catboost_baseline.csv"
    save_table(path, submission)
    return path


def build_baseline_report(
    cv_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    subgroup_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    families: dict[str, list[str]],
    best_experiment: str,
    submission_path: Path,
) -> None:
    css = """
    body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}
    h1,h2{color:#1f3b4d} table{border-collapse:collapse;font-size:12px;margin:12px 0 24px}
    th,td{border:1px solid #ddd;padding:6px 8px;text-align:left} th{background:#f2f5f7}
    .note{background:#fff8dc;border-left:4px solid #c99700;padding:10px 12px}
    """

    def table(df: pd.DataFrame, n: int = 20) -> str:
        return df.head(n).to_html(index=False, escape=True, border=0)

    best_row = ablation_results[ablation_results["experiment"].eq(best_experiment)].iloc[0]
    content = [
        f"<style>{css}</style>",
        "<h1>CatBoost Baseline Modeling Report</h1>",
        f'<p class="note">{html.escape(LEAKAGE_NOTE)}</p>',
        "<h2>Run Summary</h2>",
        f"<p>Best experiment: {html.escape(best_experiment)} / mean CV AUC: {best_row['mean_auc']:.6f}</p>",
        f"<p>Submission candidate: {html.escape(str(submission_path))}</p>",
        "<h2>Leakage-safe Modeling Principles</h2>",
        f"<p>{html.escape(LEAKAGE_NOTE)} No target encoding, no one-hot fitting on train+test, no test EDA.</p>",
        "<h2>Feature Families</h2>",
        table(pd.DataFrame({"family": families.keys(), "feature_count": [len(v) for v in families.values()]}), 50),
        "<h2>CV Results</h2>",
        table(cv_results, 50),
        "<h2>Ablation Results</h2>",
        table(ablation_results, 20),
        "<h2>Best Experiment</h2>",
        table(pd.DataFrame([best_row]), 1),
        "<h2>Subgroup AUC</h2>",
        table(subgroup_results.sort_values("auc", ascending=True, na_position="last"), 50),
        "<h2>Top 50 Feature Importance</h2>",
        table(feature_importance, 50),
        "<h2>Interpretation</h2>",
        "<p>Compare ablation rows to judge branch flags, embryo reason, transfer/day5/no-transfer, age-source interaction, and funnel-ratio contribution. Subgroup rows identify weak pockets by low or undefined AUC.</p>",
        "<h2>Next Step Recommendation</h2>",
        "<p>Use this candidate as a sanity baseline only. Next work should broaden full ablation if time allows, inspect fold stability, and then tune CatBoost or LightGBM without test-distribution feedback.</p>",
    ]
    (OUTPUT_DIR / "baseline_report.html").write_text("\n".join(content), encoding="utf-8")


def make_modeling_log(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    ablation_results: pd.DataFrame,
    best_experiment: str,
    submission_path: Path,
    warnings_list: list[str],
) -> None:
    import catboost
    import lightgbm
    import sklearn

    lines = [
        f"run_time: {datetime.now().isoformat(timespec='seconds')}",
        f"python_version: {platform.python_version()}",
        f"pandas_version: {pd.__version__}",
        f"numpy_version: {np.__version__}",
        f"sklearn_version: {sklearn.__version__}",
        f"catboost_version: {catboost.__version__}",
        f"lightgbm_version: {lightgbm.__version__}",
        f"train_path: {TRAIN_PATH}",
        f"test_path: {TEST_PATH}",
        f"train_shape: {train.shape}",
        f"test_shape_schema_prediction_only: {test.shape}",
        f"target_column: {target_col}",
        "feature_generation_function: src/features/art_features.py::make_art_features",
        f"experiments: {','.join(MODEL_CONFIG['ablation_experiments_run'])}",
    ]
    for _, row in ablation_results.iterrows():
        lines.append(f"experiment_mean_auc.{row['experiment']}: {row['mean_auc']:.6f}")
    lines.extend(
        [
            f"best_experiment: {best_experiment}",
            f"best_cv_auc: {ablation_results.loc[ablation_results['experiment'].eq(best_experiment), 'mean_auc'].iloc[0]:.6f}",
            f"submission_candidate_path: {submission_path}",
            f"leakage_note: {LEAKAGE_NOTE}",
            f"submission_note: {SUBMISSION_NOTE}",
            "warnings:",
        ]
    )
    lines.extend(f"- {w}" for w in warnings_list) if warnings_list else lines.append("- none")
    (OUTPUT_DIR / "modeling_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings_list: list[str] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        target_col = get_target_column(train)
        y = train[target_col].astype(int)

        X = make_art_features(train)
        X_test = make_art_features(test)
        families = save_feature_families(OUTPUT_DIR / "feature_list_by_family.json", X.columns.tolist())
        save_json(OUTPUT_DIR / "model_config.json", MODEL_CONFIG)

        cv_results, oof_all, model_map, importance_map = run_ablation_suite(X, y, families)
        ablation_results = summarize_ablation(cv_results)
        best_experiment = best_experiment_from(ablation_results)
        oof_best = attach_ids_to_oof(oof_all, train, best_experiment)

        best_families = dict(EXPERIMENTS)[best_experiment]
        best_features = feature_columns_for_experiment(families, best_families)
        feature_importance = build_feature_importance(importance_map[best_experiment], families)
        subgroup_results = make_subgroup_auc_results(X, train, oof_best)
        final_model, _ = train_final_model(X, y, best_features)
        submission_path = create_submission_candidate(final_model, X_test, best_features, test)

        save_table(OUTPUT_DIR / "cv_results.csv", cv_results)
        save_table(OUTPUT_DIR / "ablation_results.csv", ablation_results)
        save_table(OUTPUT_DIR / "oof_predictions.csv", oof_best)
        save_table(OUTPUT_DIR / "subgroup_auc_results.csv", subgroup_results)
        save_table(OUTPUT_DIR / "feature_importance_catboost.csv", feature_importance)
        build_baseline_report(cv_results, ablation_results, subgroup_results, feature_importance, families, best_experiment, submission_path)

        warnings_list.extend(str(w.message) for w in caught)
        make_modeling_log(train, test, target_col, ablation_results, best_experiment, submission_path, warnings_list)


if __name__ == "__main__":
    main()
