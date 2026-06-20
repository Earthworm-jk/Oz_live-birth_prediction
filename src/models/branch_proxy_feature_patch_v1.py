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
from sklearn.model_selection import StratifiedKFold

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.features.branch_proxy_features import (
    add_branch_proxy_features,
    get_patch_feature_metadata,
    patch_feature_names,
)
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup
from src.models.model_branch_diagnostics_v1 import (
    LGBM_CONFIG,
    categorical_columns,
    load_catboost_oof,
    make_blends,
    run_lgbm_cv,
    subgroup_auc_by_model,
)
from src.models.model_utils import compute_auc, get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
OUT_DIR = PROJECT_ROOT / "outputs" / "branch_proxy_feature_patch_v1"
OPTIONAL_SUB_DIR = OUT_DIR / "optional_submissions"
RANDOM_SEED = 42
N_SPLITS = 5
BASELINE_CB_NAME = "CB_long_all_features_depth6"
PATCH_CB_NAME = "CB_patch_branch_proxy_depth6"
PATCH_LGBM_NAME = "LGBM_patch_branch_proxy"
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, target encoding, train+test concat, test EDA, "
    "test-wide post-processing, and any use of test data for training or feature decisions were not used."
)

CATBOOST_PATCH_CONFIG = {
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

PATCH_GROUP_ORDER = [
    "frozen_proxy",
    "donor_egg_proxy",
    "transfer_positive_proxy",
    "day5_proxy",
    "oocyte_embryo_nonlinear_proxy",
    "low_probability_branch_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Branch-Proxy Feature Patch v1")
    parser.add_argument("--quick", action="store_true", help="Use 3 folds and shorter boosting for smoke verification.")
    parser.add_argument("--make-optional-submissions", action="store_true", help="Create candidates only when OOF criteria are met.")
    return parser.parse_args()


def unique_existing(cols: list[str], X: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in X.columns]


def add_patch_families(families: dict[str, list[str]], metadata: pd.DataFrame) -> dict[str, list[str]]:
    patched = {k: list(v) for k, v in families.items()}
    if metadata.empty:
        return patched
    patch_cols = set(metadata["feature_name"].astype(str))
    patched["raw_base"] = [c for c in patched.get("raw_base", []) if c not in patch_cols]
    for group, sub in metadata.groupby("feature_group"):
        patched[f"patch_{group}"] = sub["feature_name"].astype(str).tolist()
    return patched


def all_feature_columns(families: dict[str, list[str]], X: pd.DataFrame) -> list[str]:
    return unique_existing(all_cols_from_families(families, X), X)


def make_patch_feature_sets(X: pd.DataFrame, families: dict[str, list[str]], metadata: pd.DataFrame) -> dict[str, list[str]]:
    all_cols = all_feature_columns(families, X)
    patch_cols = patch_feature_names(X)
    without_patch = [c for c in all_cols if c not in set(patch_cols)]
    sets = {
        "baseline_rebuilt_without_patch": without_patch,
        "patch_all_branch_proxy": all_cols,
    }
    for group in PATCH_GROUP_ORDER:
        group_cols = metadata.loc[metadata["feature_group"].eq(group), "feature_name"].astype(str).tolist()
        sets[f"without_{group}"] = [c for c in all_cols if c not in set(group_cols)]
        sets[f"only_{group}_added"] = unique_existing(without_patch + group_cols, X)
    return {k: unique_existing(v, X) for k, v in sets.items()}


def run_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    model_name: str,
    feature_set: str,
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = unique_existing(feature_cols, X)
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.Series] = []
    start = time.time()

    from catboost import CatBoostClassifier

    for fold, (tr, va) in enumerate(cv.split(X[feature_cols], y), start=1):
        X_tr, X_va = X.iloc[tr][feature_cols].copy(), X.iloc[va][feature_cols].copy()
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        X_tr, cat_cols = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model = CatBoostClassifier(**config)
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict_proba(X_va)[:, 1]
        auc = compute_auc(y_va, pred)
        best_iter = int(model.get_best_iteration() or config["iterations"])
        fold_aucs.append(auc)
        best_iterations.append(best_iter)
        oof_parts.append(
            pd.DataFrame(
                {
                    "ID": X.index[va].astype(str),
                    "y_true": y_va.to_numpy(),
                    "fold": fold,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "oof_pred": pred,
                }
            )
        )
        importance_parts.append(pd.Series(model.get_feature_importance(), index=feature_cols, name=f"fold_{fold}"))

    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importance_parts, axis=1)
    cat_cols = categorical_columns(X[feature_cols])
    row = {
        "model_name": model_name,
        "model_type": "CatBoost",
        "feature_set": feature_set,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(cat_cols),
        "numeric_feature_count": len(feature_cols) - len(cat_cols),
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "training_time_sec": round(time.time() - start, 3),
        "note": "Branch-proxy patch CatBoost CV",
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
                }
            )
    return pd.DataFrame(rows)


def run_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
    patch_full_auc: float | None = None,
) -> pd.DataFrame:
    rows = []
    for feature_set in ["baseline_rebuilt_without_patch"] + [f"without_{g}" for g in PATCH_GROUP_ORDER] + [f"only_{g}_added" for g in PATCH_GROUP_ORDER]:
        if feature_set not in feature_sets:
            continue
        row, _, _ = run_catboost_cv(
            X,
            y,
            feature_sets[feature_set],
            f"CB_ablation_{feature_set}",
            feature_set,
            config,
            n_splits,
            lookup,
        )
        rows.append(row)
    ablation = pd.DataFrame(rows)
    if not ablation.empty:
        base_auc = ablation.loc[ablation["feature_set"].eq("baseline_rebuilt_without_patch"), "oof_auc"]
        ablation["delta_vs_rebuilt_baseline"] = ablation["oof_auc"] - (base_auc.iloc[0] if len(base_auc) else np.nan)
        ablation["delta_vs_patch_full"] = ablation["oof_auc"] - (patch_full_auc if patch_full_auc is not None else np.nan)
    return ablation


def branch_focus_diagnostics(subgroup: pd.DataFrame) -> pd.DataFrame:
    branch_map = {
        "overall": ("overall", "all"),
        "frozen": ("is_frozen_embryo", "1"),
        "donor_egg": ("is_donor_egg", "1"),
        "transfer_positive": ("embryo_transferred_flag", "1"),
        "day5_transfer": ("transfer_day_5", "1"),
        "DI": ("treatment_type", "DI"),
        "storage_only": ("storage_only_flag", "1"),
        "donation_only": ("donation_only_flag", "1"),
        "no_transfer": ("no_embryo_transfer_flag", "1"),
    }
    rows = []
    for branch, (subgroup_name, value) in branch_map.items():
        sub = subgroup[subgroup["subgroup"].eq(subgroup_name) & subgroup["value"].astype(str).eq(value)]
        if sub.empty:
            continue
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        baseline = sub[sub["model_name"].eq(BASELINE_CB_NAME)]
        patch_cb = sub[sub["model_name"].eq(PATCH_CB_NAME)]
        patch_lgbm = sub[sub["model_name"].eq(PATCH_LGBM_NAME)]
        blends = sub[sub["model_name"].str.startswith("blend", na=False)]
        baseline_auc = baseline["auc"].iloc[0] if len(baseline) else np.nan
        patch_cb_auc = patch_cb["auc"].iloc[0] if len(patch_cb) else np.nan
        patch_lgbm_auc = patch_lgbm["auc"].iloc[0] if len(patch_lgbm) else np.nan
        blend_auc = blends["auc"].max() if len(blends) else np.nan
        rows.append(
            {
                "branch_name": branch,
                "n": int(best["n"]),
                "positive_rate": best["positive_rate"],
                "best_model_by_auc": best["model_name"],
                "best_auc": best["auc"],
                "baseline_catboost_auc": baseline_auc,
                "patch_catboost_auc": patch_cb_auc,
                "patch_lgbm_auc": patch_lgbm_auc,
                "best_blend_auc": blend_auc,
                "delta_patch_cb_vs_baseline": patch_cb_auc - baseline_auc if pd.notna(patch_cb_auc) and pd.notna(baseline_auc) else np.nan,
                "delta_patch_lgbm_vs_baseline": patch_lgbm_auc - baseline_auc if pd.notna(patch_lgbm_auc) and pd.notna(baseline_auc) else np.nan,
                "delta_best_vs_baseline": best["auc"] - baseline_auc if pd.notna(best["auc"]) and pd.notna(baseline_auc) else np.nan,
                "interpretation": "patch helps branch" if pd.notna(patch_cb_auc) and pd.notna(baseline_auc) and patch_cb_auc > baseline_auc else "monitor",
            }
        )
    return pd.DataFrame(rows)


def optional_submission_allowed(cv_results: pd.DataFrame, blends: pd.DataFrame) -> tuple[bool, str]:
    baseline_auc = cv_results.loc[cv_results["model_name"].eq(BASELINE_CB_NAME), "oof_auc"]
    patch_auc = cv_results.loc[cv_results["model_name"].eq(PATCH_CB_NAME), "oof_auc"]
    best_blend_auc = blends["oof_auc"].max() if not blends.empty else np.nan
    if baseline_auc.empty or patch_auc.empty:
        return False, "missing baseline or patch CV"
    baseline = float(baseline_auc.iloc[0])
    patch = float(patch_auc.iloc[0])
    best = max(patch, float(best_blend_auc) if pd.notna(best_blend_auc) else -np.inf)
    if best >= baseline + 0.0002:
        return True, f"OOF improvement met threshold: best {best:.6f} vs baseline {baseline:.6f}"
    return False, f"OOF improvement below threshold: best {best:.6f} vs baseline {baseline:.6f}"


def build_report(tables: dict[str, pd.DataFrame], optional_note: str) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def table(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    cv = tables["patch_cv_results"].sort_values("oof_auc", ascending=False)
    best = cv.iloc[0] if not cv.empty else pd.Series({"model_name": "none", "oof_auc": np.nan})
    parts = [
        f"<style>{css}</style>",
        "<h1>Branch-Proxy Feature Patch v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best diagnostic model: {html.escape(str(best['model_name']))}, OOF AUC {best['oof_auc']:.6f}. This is a feature patch diagnostic stage.</p>",
        "<h2>Patch Feature List</h2>",
        table("patch_feature_list", 80),
        "<h2>CV Results</h2>",
        table("patch_cv_results", 20),
        "<h2>Branch Focus Diagnostics</h2>",
        table("patch_branch_focus_diagnostics", 30),
        "<h2>Subgroup AUC</h2>",
        table("patch_subgroup_auc", 60),
        "<h2>Patch Ablation</h2>",
        table("patch_ablation_results", 30),
        "<h2>OOF Correlation</h2>",
        table("patch_oof_correlation", 30),
        "<h2>Diagnostic Blend Results</h2>",
        table("patch_diagnostic_blend_results", 30),
        "<h2>Feature Importance</h2>",
        table("patch_feature_importance", 80),
        "<h2>Optional Submission Status</h2>",
        f"<p>{html.escape(optional_note)}</p>",
    ]
    (OUT_DIR / "branch_proxy_feature_patch_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []
    n_splits = 3 if args.quick else N_SPLITS
    cb_config = dict(CATBOOST_PATCH_CONFIG)
    lgbm_config = dict(LGBM_CONFIG)
    if args.quick:
        cb_config.update({"iterations": 180, "early_stopping_rounds": 60})
        lgbm_config.update({"n_estimators": 300})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        train = pd.read_csv(TRAIN_PATH)
        target = get_target_column(train)
        y = train[target].astype(int)
        X_base = make_art_features(train)
        X = add_branch_proxy_features(X_base)
        X.index = train[ID_COLUMN].astype(str)
        metadata = get_patch_feature_metadata(X)
        base_families = save_feature_families(OUT_DIR / "feature_families_base_plus_patch_raw.json", X.columns.tolist())
        families = add_patch_families(base_families, metadata)
        save_json(OUT_DIR / "feature_families_patch.json", families)
        lookup = family_lookup(families)
        feature_sets = make_patch_feature_sets(X, families, metadata)
        save_table(OUT_DIR / "patch_feature_list.csv", metadata)
        save_json(
            OUT_DIR / "model_config.json",
            {
                "cv": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
                "catboost_patch": cb_config,
                "lightgbm_patch": lgbm_config,
                "baseline_oof_model": BASELINE_CB_NAME,
                "patch_feature_count": len(metadata),
                "prohibited_methods": PROHIBITED_NOTE,
                "quick_mode": args.quick,
            },
        )

        oof_frames: list[pd.DataFrame] = []
        result_frames: list[pd.DataFrame] = []
        importance_frames: list[pd.DataFrame] = []

        cb_base_oof, cb_base_result = load_catboost_oof(train)
        oof_frames.append(cb_base_oof)
        result_frames.append(cb_base_result)
        completed.append(f"{BASELINE_CB_NAME}_loaded")

        patch_cols = feature_sets["patch_all_branch_proxy"]
        row, oof, imp = run_catboost_cv(X, y, patch_cols, PATCH_CB_NAME, "patch_all_branch_proxy", cb_config, n_splits, lookup)
        result_frames.append(pd.DataFrame([row]))
        oof_frames.append(oof)
        importance_frames.append(imp)
        completed.append(PATCH_CB_NAME)

        lgbm_row, lgbm_oof, lgbm_imp = run_lgbm_cv(
            X,
            y,
            patch_cols,
            PATCH_LGBM_NAME,
            "patch_all_branch_proxy",
            lgbm_config,
            n_splits,
            lookup,
        )
        result_frames.append(pd.DataFrame([lgbm_row]))
        if not lgbm_oof.empty:
            oof_frames.append(lgbm_oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]])
            importance_frames.append(lgbm_imp)
        completed.append(PATCH_LGBM_NAME)

        ablation_config = dict(cb_config)
        if not args.quick:
            ablation_config.update({"iterations": 300, "early_stopping_rounds": 80})
        ablation = run_ablation(X, y, feature_sets, ablation_config, n_splits, lookup, row["oof_auc"])
        completed.append("patch_ablation")

        cv_results = pd.concat(result_frames, ignore_index=True).sort_values("oof_auc", ascending=False)
        oof_all = pd.concat(oof_frames, ignore_index=True)
        importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        corr = correlation_table(oof_all)
        blends = make_blends(oof_all, cv_results, corr)
        subgroup = subgroup_auc_by_model(X, train, oof_all, blends)
        branch = branch_focus_diagnostics(subgroup)

        allowed, optional_note = optional_submission_allowed(cv_results, blends)
        if args.make_optional_submissions and allowed:
            skipped.append("optional_submission_generation: criteria met but test fit is intentionally left for final candidate stage")
            optional_note += "; optional generation not implemented in diagnostics-only script"
        elif args.make_optional_submissions:
            skipped.append(f"optional_submission_generation: {optional_note}")
        else:
            skipped.append("optional_submission_generation: disabled by default")
            optional_note = "Optional submissions disabled by default; " + optional_note

        tables = {
            "patch_feature_list": metadata,
            "patch_cv_results": cv_results,
            "patch_oof_predictions": oof_all,
            "patch_subgroup_auc": subgroup,
            "patch_branch_focus_diagnostics": branch,
            "patch_feature_importance": importance,
            "patch_ablation_results": ablation,
            "patch_oof_correlation": corr,
            "patch_diagnostic_blend_results": blends,
        }
        output_names = {
            "patch_cv_results": "patch_cv_results.csv",
            "patch_oof_predictions": "patch_oof_predictions.csv",
            "patch_subgroup_auc": "patch_subgroup_auc.csv",
            "patch_branch_focus_diagnostics": "patch_branch_focus_diagnostics.csv",
            "patch_feature_importance": "patch_feature_importance.csv",
            "patch_ablation_results": "patch_ablation_results.csv",
            "patch_oof_correlation": "patch_oof_correlation.csv",
            "patch_diagnostic_blend_results": "patch_diagnostic_blend_results.csv",
        }
        for key, filename in output_names.items():
            save_table(OUT_DIR / filename, tables[key])
        build_report(tables, optional_note)

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import lightgbm
        import sklearn

        best = cv_results.iloc[0]
        baseline_auc = cv_results.loc[cv_results["model_name"].eq(BASELINE_CB_NAME), "oof_auc"].iloc[0]
        patch_auc = cv_results.loc[cv_results["model_name"].eq(PATCH_CB_NAME), "oof_auc"].iloc[0]
        best_blend_auc = blends["oof_auc"].max() if not blends.empty else np.nan
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
            f"target_column: {target}",
            f"cv_setting: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"patch_feature_count: {len(metadata)}",
            f"best_model: {best['model_name']}",
            f"best_oof_auc: {best['oof_auc']:.6f}",
            f"baseline_catboost_oof_auc: {baseline_auc:.6f}",
            f"patch_catboost_oof_auc: {patch_auc:.6f}",
            f"delta_patch_catboost_vs_baseline: {patch_auc - baseline_auc:.6f}",
            f"best_diagnostic_blend_oof_auc: {best_blend_auc if pd.notna(best_blend_auc) else np.nan}",
            f"optional_submission_status: {optional_note}",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "branch_proxy_feature_patch_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
