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

from src.features.art_features import ID_COLUMN, get_feature_families, make_art_features, save_feature_families
from src.models.model_utils import compute_auc, get_categorical_columns, get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CANDIDATES = [
    RAW_DIR / "sample_submission.csv",
    PROJECT_ROOT / "submissions" / "sample_submission.csv",
]
OUT_DIR = PROJECT_ROOT / "outputs" / "controlled_full_cv_v1"
SUBMISSION_DIR = OUT_DIR / "submissions"
RANDOM_SEED = 42
N_SPLITS = 5

PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, test-wide post-processing, and any use of test data "
    "for training or feature decisions are prohibited and were not used."
)

CATBOOST_CONFIG = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 150,
    "learning_rate": 0.055,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "early_stopping_rounds": 60,
    "allow_writing_files": False,
}
CONFIG_NOTE = (
    "Controlled full CV uses 5 folds. CatBoost iterations were reduced from the 2000 draft "
    "to 150 in this workspace after a long-run timeout to complete the controlled comparison."
)

GLOBAL_RATIOS = [
    "embryo_creation_rate",
    "icsi_embryo_creation_rate",
    "transfer_per_embryo_rate",
    "storage_per_embryo_rate",
    "partner_sperm_mix_share",
    "donor_sperm_mix_share",
    "thawed_embryo_transfer_proxy",
]
SAFE_RATIOS = [
    "embryo_creation_rate",
    "storage_per_embryo_rate",
    "partner_sperm_mix_share",
    "donor_sperm_mix_share",
    "transfer_gt_created_flag",
    "fresh_current_funnel_valid_flag",
    "frozen_funnel_flag",
    "icsi_funnel_flag",
]
RAW_TRANSFER_KEEP = ["이식된 배아 수", "배아 이식 경과일", "단일 배아 이식 여부"]
TRANSFER_MINIMAL_KEEP = [
    "이식된 배아 수",
    "배아 이식 경과일",
    "단일 배아 이식 여부",
    "no_embryo_transfer_flag",
    "embryo_transferred_flag",
    "transfer_day_missing",
    "transfer_day_5",
    "transfer_day_ge5",
    "possible_blastocyst_transfer",
]


def make_model():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**CATBOOST_CONFIG)


def unique_existing(cols: list[str], X: pd.DataFrame) -> list[str]:
    return [c for c in dict.fromkeys(cols) if c in X.columns]


def all_cols_from_families(families: dict[str, list[str]], X: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for family_cols in families.values():
        cols.extend(family_cols)
    return unique_existing(cols, X)


def family_lookup(families: dict[str, list[str]]) -> dict[str, str]:
    lookup = {}
    for family, cols in families.items():
        for col in cols:
            lookup[col] = family
    return lookup


def build_feature_sets(X: pd.DataFrame, families: dict[str, list[str]]) -> dict[str, list[str]]:
    all_cols = all_cols_from_families(families, X)
    transfer_family = set(families.get("transfer_features", []))
    transfer_dupes = {c for c in all_cols if "embryo_transferred_count" in c}
    minimal = [c for c in all_cols if c not in transfer_family and c not in transfer_dupes]
    minimal += TRANSFER_MINIMAL_KEEP

    no_global_ratio = [c for c in all_cols if c not in set(GLOBAL_RATIOS)]
    safe_ratio = [c for c in all_cols if c not in set(GLOBAL_RATIOS)] + SAFE_RATIOS
    selected = no_global_ratio
    if len(set(minimal)) < len(all_cols):
        selected = [c for c in no_global_ratio if c not in transfer_family and c not in transfer_dupes] + TRANSFER_MINIMAL_KEEP
    return {
        "all_features_expanded_transfer": unique_existing(all_cols, X),
        "all_features_transfer_minimal": unique_existing(minimal, X),
        "all_features_no_global_ratio": unique_existing(no_global_ratio, X),
        "all_features_safe_ratio_only": unique_existing(safe_ratio, X),
        "selected_features_v1": unique_existing(selected, X),
    }


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    experiment: str,
    subset_mask: pd.Series | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = unique_existing(feature_cols, X)
    if subset_mask is None:
        subset_mask = pd.Series(True, index=X.index)
    subset_mask = subset_mask.fillna(False)
    row_positions = np.flatnonzero(subset_mask.to_numpy())
    X_sub = X.iloc[row_positions][feature_cols].reset_index(drop=True)
    y_sub = y.iloc[row_positions].reset_index(drop=True)

    if len(row_positions) == 0 or y_sub.nunique() < 2:
        return (
            {
                "experiment": experiment,
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
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    n_splits = min(N_SPLITS, int(y_sub.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    importances: list[pd.Series] = []
    cat_cols = get_categorical_columns(X_sub)
    num_cols = [c for c in X_sub.columns if c not in cat_cols]
    start = time.time()

    for fold, (tr, va) in enumerate(cv.split(X_sub, y_sub), start=1):
        X_tr, X_va = X_sub.iloc[tr].copy(), X_sub.iloc[va].copy()
        y_tr, y_va = y_sub.iloc[tr], y_sub.iloc[va]
        X_tr, cat_fit = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_fit]
        model = make_model()
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict_proba(X_va)[:, 1]
        fold_aucs.append(compute_auc(y_va, pred))
        best_iterations.append(int(model.get_best_iteration() or CATBOOST_CONFIG["iterations"]))
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
        importances.append(pd.Series(model.get_feature_importance(), index=feature_cols, name=f"fold_{fold}"))

    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(importances, axis=1)
    row = {
        "experiment": experiment,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(cat_cols),
        "numeric_feature_count": len(num_cols),
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "training_time_sec": round(time.time() - start, 3),
        "note": CONFIG_NOTE,
    }
    return row, oof, imp


def importance_table(experiment: str, imp: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "experiment": experiment,
            "feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "feature_family": [lookup.get(c, "unknown") for c in imp.index],
        }
    ).sort_values("importance_mean", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def subgroup_auc_table(X: pd.DataFrame, raw: pd.DataFrame, oof_all: pd.DataFrame) -> pd.DataFrame:
    frame = X.copy()
    frame["row_index"] = np.arange(len(frame))
    merged = oof_all.merge(frame, on="row_index", how="left")
    specs: list[tuple[str, str, pd.Series]] = []
    specs.append(("overall", "all", pd.Series(True, index=merged.index)))
    for col, name in [("시술 유형", "treatment_type"), ("age_group_raw", "age_group"), ("egg_source_raw", "egg_source")]:
        if col in merged:
            for value in sorted(merged[col].dropna().astype(str).unique()):
                specs.append((name, value, merged[col].astype(str).eq(value)))
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
        if col in merged:
            specs.append((col, "1", merged[col].eq(1)))

    rows = []
    for name, value, mask in specs:
        sub = merged[mask.fillna(False)]
        if sub.empty:
            continue
        y_true = sub["y_true"]
        pred = sub["oof_pred"]
        pos = pred[y_true.eq(1)].mean() if y_true.eq(1).any() else np.nan
        neg = pred[y_true.eq(0)].mean() if y_true.eq(0).any() else np.nan
        auc = compute_auc(y_true, pred) if y_true.nunique() > 1 else np.nan
        rows.append(
            {
                "experiment": sub["experiment"].iloc[0],
                "subgroup": name,
                "value": value,
                "n": len(sub),
                "positive_rate": y_true.mean(),
                "auc": auc,
                "pred_mean": pred.mean(),
                "calibration_gap": pred.mean() - y_true.mean(),
                "separation": pos - neg if pd.notna(pos) and pd.notna(neg) else np.nan,
                "note": "" if y_true.nunique() > 1 else "only one class present",
            }
        )
    return pd.DataFrame(rows)


def train_final_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    feature_cols = unique_existing(feature_cols, X)
    X_fit, cat_cols = prepare_catboost_frame(X[feature_cols].copy())
    X_pred, _ = prepare_catboost_frame(X_test[feature_cols].copy())
    cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols]
    model = make_model()
    model.fit(X_fit, y, cat_features=cat_idx)
    return model.predict_proba(X_pred)[:, 1]


def create_submission(path: Path, sample: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != ID_COLUMN][0]
    sub[pred_col] = preds
    save_table(path, sub)
    return sub


def sanity_rows(sample: pd.DataFrame, submissions: dict[str, tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    sample_id = sample[ID_COLUMN].astype(str)
    for name, (path, sub) in submissions.items():
        pred_col = [c for c in sub.columns if c != ID_COLUMN][0]
        pred = sub[pred_col]
        rows.append(
            {
                "candidate": name,
                "path": str(path),
                "row_count_equals_sample": len(sub) == len(sample),
                "id_order_matches_sample": sub[ID_COLUMN].astype(str).equals(sample_id),
                "prediction_has_no_nan": pred.notna().all(),
                "prediction_min": pred.min(),
                "prediction_max": pred.max(),
                "prediction_mean": pred.mean(),
                "prediction_std": pred.std(),
                "duplicated_id_count": int(sub[ID_COLUMN].duplicated().sum()),
                "note": "submission candidate only, not submitted",
            }
        )
    return pd.DataFrame(rows)


def blend_results(best_oof: pd.DataFrame, X: pd.DataFrame, frozen_oof: pd.DataFrame, di_oof: pd.DataFrame) -> pd.DataFrame:
    base = best_oof.copy()
    base["blend_pred"] = base["oof_pred"]
    global_auc = compute_auc(base["y_true"], base["oof_pred"])
    rows = [
        {
            "blend_name": "global",
            "overall_oof_auc": global_auc,
            "delta_vs_global_oof_auc": 0.0,
            "affected_rows": 0,
            "affected_positive_rate": np.nan,
            "note": "best global OOF",
        }
    ]

    masks = {
        "blend_frozen_only": X["is_frozen_embryo"].eq(1),
        "blend_di_only": X["is_di"].eq(1),
        "blend_frozen_di": X["is_frozen_embryo"].eq(1) | X["is_di"].eq(1),
    }
    specialist_map = {
        "blend_frozen_only": [frozen_oof],
        "blend_di_only": [di_oof],
        "blend_frozen_di": [frozen_oof, di_oof],
    }
    for name, mask in masks.items():
        blended = base.copy()
        for spec in specialist_map[name]:
            if spec.empty:
                continue
            spec_map = spec.set_index("ID")["oof_pred"]
            update = blended["ID"].isin(spec_map.index)
            blended.loc[update, "blend_pred"] = blended.loc[update, "ID"].map(spec_map)
        auc = compute_auc(blended["y_true"], blended["blend_pred"])
        affected_ids = X.index[mask].astype(str)
        affected = blended[blended["ID"].isin(affected_ids)]
        rows.append(
            {
                "blend_name": name,
                "overall_oof_auc": auc,
                "delta_vs_global_oof_auc": auc - global_auc,
                "affected_rows": len(affected),
                "affected_positive_rate": affected["y_true"].mean() if len(affected) else np.nan,
                "note": "OOF-only specialist blend validation; not applied to test in this stage",
            }
        )
    return pd.DataFrame(rows)


def build_report(tables: dict[str, pd.DataFrame], best_exp: str, paths: list[Path]) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def t(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    best_row = tables["full_cv_results"].query("experiment == @best_exp").iloc[0]
    parts = [
        f"<style>{css}</style>",
        "<h1>Controlled Full CV v1 Report</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best global experiment: {html.escape(best_exp)} / OOF AUC {best_row['oof_auc']:.6f}.</p>",
        "<h2>Leakage and Prohibited Methods</h2>",
        "<p>Pseudo-labeling was not used. Test-wide post-processing was not used. Test was used only for row-wise transformation and submission candidate prediction. No automatic submission was made.</p>",
        "<h2>Full CV Setup</h2>",
        f"<p>5-fold StratifiedKFold, CatBoost config: {html.escape(str(CATBOOST_CONFIG))}. {html.escape(CONFIG_NOTE)}</p>",
        "<h2>Feature Set Comparison</h2>", t("full_cv_results", 20),
        "<h2>Transfer Variant Comparison</h2>", t("transfer_variant_comparison", 20),
        "<h2>Funnel Ratio Variant Comparison</h2>", t("funnel_ratio_variant_comparison", 20),
        "<h2>Subgroup AUC</h2>", t("subgroup_auc_full_cv", 50),
        "<h2>Frozen/DI Specialist OOF Blending</h2>", t("specialist_blend_oof_results", 20),
        "<h2>Feature Importance</h2>", t("full_cv_feature_importance", 50),
        "<h2>Submission Candidate Sanity Check</h2>", t("submission_sanity_check", 20),
        "<h2>Recommendation: Submit or Continue</h2>",
        "<p>Use the best stable global candidate for a first controlled submission if sanity checks pass. Specialist blends remain OOF-only and should be validated before test use.</p>",
        "<h2>Next Experiment Suggestions</h2>",
        "<p>Run longer CatBoost on selected/no-global-ratio, compare LightGBM, then test specialist blending in a separate leakage-safe stage.</p>",
        "<h2>Generated Candidate Paths</h2>",
        "<ul>" + "".join(f"<li>{html.escape(str(p))}</li>" for p in paths) + "</ul>",
    ]
    (OUT_DIR / "controlled_full_cv_report.html").write_text("\n".join(parts), encoding="utf-8")


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
        test = pd.read_csv(TEST_PATH)
        target = get_target_column(train)
        y = train[target].astype(int)
        X = make_art_features(train)
        X.index = train[ID_COLUMN].astype(str)
        X_test = make_art_features(test)
        X_test.index = test[ID_COLUMN].astype(str)
        families = save_feature_families(OUT_DIR / "selected_feature_sets_families.json", X.columns.tolist())
        feature_sets = build_feature_sets(X, families)
        lookup = family_lookup(families)
        save_json(OUT_DIR / "model_config.json", {"catboost": CATBOOST_CONFIG, "cv": f"StratifiedKFold(n_splits={N_SPLITS})", "note": CONFIG_NOTE})
        save_json(OUT_DIR / "selected_feature_sets.json", feature_sets)

        result_rows = []
        oof_frames = []
        importance_frames = []
        imp_maps: dict[str, pd.DataFrame] = {}
        for exp, cols in feature_sets.items():
            row, oof, imp = run_cv(X, y, cols, exp)
            result_rows.append(row)
            oof_frames.append(oof[["row_index", "ID", "y_true", "fold", "experiment", "oof_pred"]])
            imp_table = importance_table(exp, imp, lookup)
            importance_frames.append(imp_table)
            imp_maps[exp] = imp
            completed.append(exp)

        results = pd.DataFrame(result_rows).sort_values("oof_auc", ascending=False)
        best_exp = str(results.iloc[0]["experiment"])
        best_oof = pd.concat(oof_frames, ignore_index=True).query("experiment == @best_exp").copy()

        frozen_row, frozen_oof, _ = run_cv(X, y, feature_sets[best_exp], "frozen_specialist", X["is_frozen_embryo"].eq(1))
        di_row, di_oof, _ = run_cv(X, y, feature_sets[best_exp], "di_specialist", X["is_di"].eq(1))
        specialist_blend = blend_results(best_oof, X, frozen_oof, di_oof)
        completed.extend(["frozen_specialist", "di_specialist"])

        oof_long = pd.concat(oof_frames, ignore_index=True)
        subgroup_tables = []
        for exp in results["experiment"]:
            subgroup_tables.append(subgroup_auc_table(X, train, oof_long.query("experiment == @exp").copy()))
        subgroup = pd.concat(subgroup_tables, ignore_index=True)
        importance = pd.concat(importance_frames, ignore_index=True)

        transfer_comp = results[results["experiment"].isin(["all_features_expanded_transfer", "all_features_transfer_minimal"])].copy()
        funnel_comp = results[results["experiment"].isin(["all_features_expanded_transfer", "all_features_no_global_ratio", "all_features_safe_ratio_only"])].copy()

        sample_path = next((p for p in SAMPLE_SUBMISSION_CANDIDATES if p.exists()), None)
        if sample_path is None:
            sample = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(str), "probability": 0.0})
        else:
            sample = pd.read_csv(sample_path)
        submission_specs = {
            "candidate_catboost_full_all_features": ("candidate_catboost_full_all_features.csv", "all_features_expanded_transfer"),
            "candidate_catboost_full_no_global_ratio": ("candidate_catboost_full_no_global_ratio.csv", "all_features_no_global_ratio"),
            "candidate_catboost_full_selected": ("candidate_catboost_full_selected.csv", "selected_features_v1"),
        }
        submissions: dict[str, tuple[Path, pd.DataFrame]] = {}
        for name, (filename, exp) in submission_specs.items():
            preds = train_final_predict(X, y, X_test, feature_sets[exp])
            path = SUBMISSION_DIR / filename
            sub = create_submission(path, sample, preds)
            submissions[name] = (path, sub)
        sanity = sanity_rows(sample, submissions)

        tables = {
            "full_cv_results": results,
            "full_cv_oof_predictions": oof_long[["ID", "y_true", "fold", "experiment", "oof_pred"]],
            "full_cv_feature_importance": importance,
            "transfer_variant_comparison": transfer_comp,
            "funnel_ratio_variant_comparison": funnel_comp,
            "specialist_blend_oof_results": specialist_blend,
            "subgroup_auc_full_cv": subgroup,
            "submission_sanity_check": sanity,
        }
        for name, df in tables.items():
            save_table(OUT_DIR / f"{name}.csv", df)
        build_report(tables, best_exp, [p for p, _ in submissions.values()])

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import sklearn

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
            f"catboost_setting: {CATBOOST_CONFIG}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"best_global_experiment: {best_exp}",
            f"best_global_oof_auc: {results.iloc[0]['oof_auc']:.6f}",
            f"best_specialist_blend_oof_auc: {specialist_blend['overall_oof_auc'].max():.6f}",
            f"generated_submission_candidate_paths: {','.join(str(p) for p, _ in submissions.values())}",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "controlled_full_cv_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
