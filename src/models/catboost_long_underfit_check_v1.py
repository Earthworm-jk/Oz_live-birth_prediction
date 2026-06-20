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
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup, subgroup_auc_table
from src.models.model_utils import compute_auc, get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
OUT_DIR = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1"
SUBMISSION_DIR = OUT_DIR / "submissions"
RANDOM_SEED = 42
N_SPLITS = 5
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, test-wide post-processing, and any use of test data "
    "for training or feature decisions were not used."
)


BASE_CONFIG = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 600,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "early_stopping_rounds": 150,
    "allow_writing_files": False,
}
CONFIG_NOTE = (
    "Long underfit check uses 600 iterations. The original 2000-iteration draft and broader "
    "depth/lr matrix were reduced after timeout; the key raw vs all_features 5-fold underfit "
    "comparison avoids the previous 150-iteration cap."
)


EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "raw_long_depth6": {
        **BASE_CONFIG,
        "iterations": 600,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "feature_set": "raw_base",
    },
    "all_features_long_depth6": {
        **BASE_CONFIG,
        "iterations": 600,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "feature_set": "all_features_expanded_transfer",
    },
    "all_features_long_depth7": {
        **BASE_CONFIG,
        "iterations": 700,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "feature_set": "all_features_expanded_transfer",
    },
    "all_features_long_lr005": {
        **BASE_CONFIG,
        "iterations": 700,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "feature_set": "all_features_expanded_transfer",
    },
    "frozen_specialist_long": {
        **BASE_CONFIG,
        "iterations": 600,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "feature_set": "all_features_expanded_transfer",
    },
}


def make_model(config: dict[str, Any]):
    from catboost import CatBoostClassifier

    params = {k: v for k, v in config.items() if k != "feature_set"}
    return CatBoostClassifier(**params)


def unique_existing(cols: list[str], X: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in X.columns]


def run_long_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    experiment: str,
    config: dict[str, Any],
    lookup: dict[str, str],
    subset_mask: pd.Series | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = unique_existing(feature_cols, X)
    if subset_mask is None:
        subset_mask = pd.Series(True, index=X.index)
    subset_mask = subset_mask.fillna(False)
    row_positions = np.flatnonzero(subset_mask.to_numpy())
    X_sub = X.iloc[row_positions][feature_cols].reset_index(drop=True)
    y_sub = y.iloc[row_positions].reset_index(drop=True)

    if len(row_positions) == 0 or y_sub.nunique() < 2:
        row = {
            "experiment": experiment,
            "feature_set": config["feature_set"],
            "iterations": config["iterations"],
            "learning_rate": config["learning_rate"],
            "depth": config["depth"],
            "l2_leaf_reg": config["l2_leaf_reg"],
            "fold_auc_list": "",
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "oof_auc": np.nan,
            "best_iteration_list": "",
            "max_iteration_reached_count": 0,
            "training_time_sec": 0.0,
            "note": "skipped: empty subset or one class",
        }
        return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cv = StratifiedKFold(n_splits=min(N_SPLITS, int(y_sub.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.Series] = []
    iter_rows: list[dict[str, Any]] = []
    start = time.time()

    for fold, (tr, va) in enumerate(cv.split(X_sub, y_sub), start=1):
        X_tr, X_va = X_sub.iloc[tr].copy(), X_sub.iloc[va].copy()
        y_tr, y_va = y_sub.iloc[tr], y_sub.iloc[va]
        X_tr, cat_cols = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model = make_model(config)
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict_proba(X_va)[:, 1]
        auc = compute_auc(y_va, pred)
        best_iter = int(model.get_best_iteration() or config["iterations"])
        hit_cap = best_iter >= int(config["iterations"] * 0.9)
        fold_aucs.append(auc)
        best_iterations.append(best_iter)
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_index": row_positions[va],
                    "ID": X.index[row_positions[va]].astype(str),
                    "y_true": y_va.to_numpy(),
                    "fold": fold,
                    "experiment": experiment,
                    "oof_pred": pred,
                }
            )
        )
        importance_parts.append(pd.Series(model.get_feature_importance(), index=feature_cols, name=f"fold_{fold}"))
        iter_rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "best_iteration": best_iter,
                "configured_iterations": config["iterations"],
                "best_iteration_ratio": best_iter / config["iterations"],
                "hit_iteration_cap": hit_cap,
                "auc": auc,
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importance_parts, axis=1)
    max_reached = int(sum(v >= int(config["iterations"] * 0.9) for v in best_iterations))
    result = {
        "experiment": experiment,
        "feature_set": config["feature_set"],
        "iterations": config["iterations"],
        "learning_rate": config["learning_rate"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "max_iteration_reached_count": max_reached,
        "training_time_sec": round(time.time() - start, 3),
        "note": CONFIG_NOTE,
    }
    importance = pd.DataFrame(
        {
            "experiment": experiment,
            "feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(c, "unknown") for c in imp.index],
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return result, oof, importance, pd.DataFrame(iter_rows)


def create_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
) -> Path:
    sample_path = next((p for p in SAMPLE_SUBMISSION_CANDIDATES if p.exists()), None)
    if sample_path is None:
        sample = pd.DataFrame({ID_COLUMN: test_raw[ID_COLUMN].astype(str), "probability": 0.0})
    else:
        sample = pd.read_csv(sample_path)
    pred_col = [c for c in sample.columns if c != ID_COLUMN][0]
    X_fit, cat_cols = prepare_catboost_frame(X[feature_cols].copy())
    X_pred, _ = prepare_catboost_frame(X_test[feature_cols].copy())
    cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols]
    model = make_model(config)
    model.fit(X_fit, y, cat_features=cat_idx)
    sample[pred_col] = model.predict_proba(X_pred)[:, 1]
    path = SUBMISSION_DIR / "candidate_catboost_long_all_features.csv"
    save_table(path, sample)
    return path


def build_report(
    results: pd.DataFrame,
    iteration_diag: pd.DataFrame,
    subgroup: pd.DataFrame,
    importance: pd.DataFrame,
    blend_rows: pd.DataFrame,
    candidate_path: Path | None,
) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def table(df: pd.DataFrame, n: int = 30) -> str:
        return df.head(n).to_html(index=False, escape=True, border=0)

    best = results.sort_values("oof_auc", ascending=False).iloc[0]
    raw = results[results["experiment"].eq("raw_long_depth6")]
    all_d6 = results[results["experiment"].eq("all_features_long_depth6")]
    raw_vs_all = ""
    if not raw.empty and not all_d6.empty:
        raw_vs_all = f"raw OOF AUC={raw.iloc[0]['oof_auc']:.6f}, all_features depth6 OOF AUC={all_d6.iloc[0]['oof_auc']:.6f}."
    cap_rate = iteration_diag["hit_iteration_cap"].mean() if not iteration_diag.empty else np.nan
    parts = [
        f"<style>{css}</style>",
        "<h1>CatBoost Long Underfit Check v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best experiment: {html.escape(str(best['experiment']))}, OOF AUC {best['oof_auc']:.6f}. {html.escape(raw_vs_all)}</p>",
        "<h2>Why this underfit check was needed</h2>",
        "<p>The previous controlled full CV hit a 150-iteration cap on most folds, so longer CatBoost runs were needed.</p>",
        "<h2>Leakage and prohibited methods</h2>",
        "<p>No pseudo-labeling, self-training, target encoding, test EDA, train+test concat, or test-wide post-processing was used.</p>",
        "<h2>Long CatBoost settings</h2>",
        f"<p>{html.escape(CONFIG_NOTE)}</p>",
        "<h2>raw vs all_features comparison</h2>",
        table(results, 20),
        "<h2>iteration diagnostics</h2>",
        f"<p>Share of folds with best_iteration_ratio >= 0.9: {cap_rate:.3f}</p>",
        table(iteration_diag, 30),
        "<h2>depth/lr comparison</h2>",
        table(results[results["experiment"].str.contains("depth7|lr005|depth6", regex=True)], 20),
        "<h2>subgroup AUC</h2>",
        table(subgroup, 50),
        "<h2>frozen blend optional result</h2>",
        table(blend_rows, 10),
        "<h2>feature importance</h2>",
        table(importance, 50),
        "<h2>Interpretation</h2>",
        "<p>If long all_features beats raw and 150-cap CV, the feature direction is acceptable and the earlier model was underfit. If most folds still hit cap, even longer training may help.</p>",
        "<h2>Recommendation: feature issue or training issue?</h2>",
        f"<p>Candidate path: {html.escape(str(candidate_path)) if candidate_path else 'not generated'}.</p>",
    ]
    (OUT_DIR / "long_underfit_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []

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
        raw_cols = families.get("raw_base", [])
        all_cols = all_cols_from_families(families, X)
        feature_sets = {
            "raw_base": raw_cols,
            "all_features_expanded_transfer": all_cols,
        }
        save_json(OUT_DIR / "model_config.json", {"experiments": EXPERIMENT_CONFIGS, "cv": f"{N_SPLITS}-fold", "note": CONFIG_NOTE})

        rows, oof_frames, imp_frames, iter_frames = [], [], [], []
        skipped.extend(["all_features_long_depth7: skipped after long-run timeout", "all_features_long_lr005: skipped after long-run timeout"])
        for exp in ["raw_long_depth6", "all_features_long_depth6"]:
            config = EXPERIMENT_CONFIGS[exp]
            result, oof, imp, it = run_long_cv(X, y, feature_sets[config["feature_set"]], exp, config, lookup)
            rows.append(result)
            oof_frames.append(oof[["ID", "y_true", "fold", "experiment", "oof_pred"]])
            imp_frames.append(imp)
            iter_frames.append(it)
            completed.append(exp)

        results = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
        oof_all = pd.concat(oof_frames, ignore_index=True)
        importance = pd.concat(imp_frames, ignore_index=True)
        iteration_diag = pd.concat(iter_frames, ignore_index=True)

        subgroup_frames = []
        for exp in results["experiment"]:
            exp_oof = oof_all[oof_all["experiment"].eq(exp)].copy()
            # subgroup_auc_table expects row_index, so reconstruct from ID.
            exp_oof = exp_oof.merge(pd.DataFrame({"ID": X.index.astype(str), "row_index": np.arange(len(X))}), on="ID", how="left")
            subgroup_frames.append(subgroup_auc_table(X, train, exp_oof))
        subgroup = pd.concat(subgroup_frames, ignore_index=True)

        blend_rows = []
        candidate_path: Path | None = None
        global_oof = oof_all[oof_all["experiment"].eq("all_features_long_depth6")].copy()
        if not global_oof.empty:
            global_auc = compute_auc(global_oof["y_true"], global_oof["oof_pred"])
            blend_rows = [
                {
                    "blend_name": "global_all_features_long_depth6",
                    "global_oof_auc": global_auc,
                    "frozen_specialist_subset_auc": np.nan,
                    "blend_oof_auc": np.nan,
                    "delta_vs_global": np.nan,
                    "affected_rows": 0,
                    "note": "Frozen specialist skipped after long-run timeout; not automatically applied to test.",
                }
            ]
            skipped.append("frozen_specialist_long: skipped after long-run timeout")

            test = pd.read_csv(TEST_PATH)
            X_test = make_art_features(test)
            X_test.index = test[ID_COLUMN].astype(str)
            best_exp = str(results.iloc[0]["experiment"])
            best_config = EXPERIMENT_CONFIGS[best_exp]
            best_cols = feature_sets[best_config["feature_set"]]
            candidate_path = create_candidate(X, y, X_test, test, best_cols, best_config)
        else:
            skipped.append("frozen_specialist_long")

        blend_df = pd.DataFrame(blend_rows)
        save_table(OUT_DIR / "long_cv_results.csv", results)
        save_table(OUT_DIR / "long_oof_predictions.csv", oof_all)
        save_table(OUT_DIR / "long_feature_importance.csv", importance)
        save_table(OUT_DIR / "subgroup_auc_long.csv", subgroup)
        save_table(OUT_DIR / "iteration_diagnostics.csv", iteration_diag)
        if not blend_df.empty:
            save_table(OUT_DIR / "frozen_blend_oof_result.csv", blend_df)
        build_report(results, iteration_diag, subgroup, importance, blend_df, candidate_path)

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import sklearn

        best = results.iloc[0]
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"sklearn_version: {sklearn.__version__}",
            f"catboost_version: {catboost.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"test_path: {TEST_PATH}",
            f"target_column: {target}",
            f"cv_setting: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
            f"catboost_setting: {EXPERIMENT_CONFIGS}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"best_experiment: {best['experiment']}",
            f"best_oof_auc: {best['oof_auc']:.6f}",
            f"best_iteration_list: {best['best_iteration_list']}",
            f"max_iteration_reached_count: {best['max_iteration_reached_count']}",
            f"generated_candidate_path: {candidate_path if candidate_path else 'none'}",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "long_underfit_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
