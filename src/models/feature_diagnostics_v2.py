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
from src.models.model_utils import compute_auc, get_categorical_columns, get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics_v2"
FIG_DIR = OUT_DIR / "figures"
RANDOM_SEED = 42
LEAKAGE_NOTE = (
    "All feature engineering is row-wise or fitted on training folds only. "
    "Test data is used only for schema-compatible transformation and final prediction, "
    "not for EDA, fitting preprocessing statistics, or feature decisions."
)


CATBOOST_CONFIG = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 30,
    "learning_rate": 0.09,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "early_stopping_rounds": 20,
    "allow_writing_files": False,
}
CV_SPLITS = 3
FAST_NOTE = "fast diagnostic mode: n_splits=3, iterations=30 to finish the full diagnostic matrix."


def make_model():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**CATBOOST_CONFIG)


def families_to_columns(families: dict[str, list[str]], family_names: list[str]) -> list[str]:
    cols: list[str] = []
    for family in family_names:
        cols.extend(families.get(family, []))
    return list(dict.fromkeys(cols))


def all_feature_columns(families: dict[str, list[str]]) -> list[str]:
    return families_to_columns(families, list(families.keys()))


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    experiment: str,
    feature_family_lookup: dict[str, str],
    subset_mask: pd.Series | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in X.columns]
    if subset_mask is None:
        subset_mask = pd.Series(True, index=X.index)
    subset_mask = subset_mask.fillna(False)
    idx = np.flatnonzero(subset_mask.to_numpy())
    y_sub = y.iloc[idx].reset_index(drop=True)
    X_sub = X.iloc[idx][feature_cols].reset_index(drop=True)

    if len(idx) == 0 or y_sub.nunique() < 2:
        row = {
            "experiment": experiment,
            "feature_count": len(feature_cols),
            "categorical_feature_count": 0,
            "numeric_feature_count": 0,
            "fold_auc_list": "",
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "oof_auc": np.nan,
            "training_time_sec": 0,
            "note": "skipped: empty subset or one class only",
        }
        return row, pd.DataFrame(), pd.DataFrame()

    skf = StratifiedKFold(n_splits=min(CV_SPLITS, y_sub.value_counts().min()), shuffle=True, random_state=RANDOM_SEED)
    oof_parts = []
    imps = []
    fold_aucs = []
    cat_cols = get_categorical_columns(X_sub)
    num_cols = [c for c in X_sub.columns if c not in cat_cols]
    start = time.time()

    for fold, (tr, va) in enumerate(skf.split(X_sub, y_sub), start=1):
        X_tr, X_va = X_sub.iloc[tr].copy(), X_sub.iloc[va].copy()
        y_tr, y_va = y_sub.iloc[tr], y_sub.iloc[va]
        X_tr, cat_cols_fit = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols_fit]
        model = make_model()
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict_proba(X_va)[:, 1]
        auc = compute_auc(y_va, pred)
        fold_aucs.append(auc)
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_index": idx[va],
                    "ID": X.index[idx[va]].astype(str),
                    "y_true": y_va.to_numpy(),
                    "experiment": experiment,
                    "fold": fold,
                    "oof_pred": pred,
                }
            )
        )
        imps.append(pd.Series(model.get_feature_importance(), index=feature_cols, name=f"fold_{fold}"))

    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(imps, axis=1)
    row = {
        "experiment": experiment,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(cat_cols),
        "numeric_feature_count": len(num_cols),
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.nanmean(fold_aucs)),
        "std_auc": float(np.nanstd(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "training_time_sec": round(time.time() - start, 3),
        "note": FAST_NOTE,
    }
    imp_out = (
        pd.DataFrame(
            {
                "experiment": experiment,
                "feature": imp.index,
                "importance_mean": imp.mean(axis=1).to_numpy(),
                "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
                "feature_family": [feature_family_lookup.get(c, "unknown") for c in imp.index],
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_out["rank"] = np.arange(1, len(imp_out) + 1)
    return row, oof, imp_out


def build_family_lookup(families: dict[str, list[str]]) -> dict[str, str]:
    lookup = {}
    for family, cols in families.items():
        for col in cols:
            lookup[col] = family
    return lookup


def family_ablation_experiments(families: dict[str, list[str]]) -> list[dict[str, Any]]:
    all_fams = list(families.keys())
    specs = [
        ("raw_only", ["raw_base"], []),
        ("raw_plus_branch", ["raw_base", "branch_flags"], []),
        ("raw_plus_treatment_tokens", ["raw_base", "treatment_tokens"], []),
        ("raw_plus_embryo_reason", ["raw_base", "embryo_reason_features"], []),
        ("raw_plus_age_source", ["raw_base", "age_source_interactions"], []),
        ("raw_plus_transfer", ["raw_base", "transfer_features"], []),
        ("raw_plus_funnel_counts", ["raw_base", "funnel_count_bin_features"], []),
        ("raw_plus_funnel_ratios", ["raw_base", "funnel_ratio_features"], []),
        ("raw_plus_history", ["raw_base", "history_features"], []),
        ("raw_plus_cause", ["raw_base", "cause_features"], []),
        ("raw_plus_branch_treatment_reason", ["raw_base", "branch_flags", "treatment_tokens", "embryo_reason_features"], []),
        ("raw_plus_age_transfer", ["raw_base", "age_source_interactions", "transfer_features"], []),
        ("raw_plus_funnel_counts_ratios", ["raw_base", "funnel_count_bin_features", "funnel_ratio_features"], []),
        ("raw_plus_history_cause", ["raw_base", "history_features", "cause_features"], []),
        ("all_features", all_fams, []),
        ("all_minus_transfer", [f for f in all_fams if f != "transfer_features"], ["transfer_features"]),
        ("all_minus_funnel_ratios", [f for f in all_fams if f != "funnel_ratio_features"], ["funnel_ratio_features"]),
        ("all_minus_funnel_counts", [f for f in all_fams if f != "funnel_count_bin_features"], ["funnel_count_bin_features"]),
        ("all_minus_history", [f for f in all_fams if f != "history_features"], ["history_features"]),
        ("all_minus_cause", [f for f in all_fams if f != "cause_features"], ["cause_features"]),
        ("all_minus_embryo_reason", [f for f in all_fams if f != "embryo_reason_features"], ["embryo_reason_features"]),
        ("all_minus_treatment_tokens", [f for f in all_fams if f != "treatment_tokens"], ["treatment_tokens"]),
        ("all_minus_branch", [f for f in all_fams if f != "branch_flags"], ["branch_flags"]),
    ]
    return [
        {
            "experiment": name,
            "feature_set_type": "single_add_or_remove",
            "included_families": ",".join(inc),
            "excluded_families": ",".join(exc),
            "feature_cols": families_to_columns(families, inc),
        }
        for name, inc, exc in specs
    ]


def transfer_experiments(families: dict[str, list[str]], all_cols: list[str]) -> list[dict[str, Any]]:
    raw_transfer = ["이식된 배아 수", "배아 이식 경과일", "단일 배아 이식 여부", "미세주입 배아 이식 수"]
    minimal = ["이식된 배아 수", "배아 이식 경과일", "no_embryo_transfer_flag", "transfer_day_missing", "transfer_day_5"]
    transfer_family = set(families.get("transfer_features", []))
    funnel_transfer_dupes = {c for c in all_cols if "embryo_transferred_count" in c}
    raw_transfer_set = set(raw_transfer)
    return [
        {"experiment": "baseline_all", "feature_cols": all_cols, "note": "all_features"},
        {"experiment": "transfer_raw_only", "feature_cols": [c for c in all_cols if c not in transfer_family and c not in funnel_transfer_dupes] + [c for c in raw_transfer if c in all_cols], "note": "raw transfer only"},
        {"experiment": "transfer_minimal", "feature_cols": [c for c in all_cols if c not in transfer_family and c not in funnel_transfer_dupes] + [c for c in minimal if c in all_cols], "note": "minimal transfer"},
        {"experiment": "transfer_expanded", "feature_cols": all_cols, "note": "expanded transfer"},
        {"experiment": "no_raw_transfer_plus_flags", "feature_cols": [c for c in all_cols if c not in raw_transfer_set], "note": "derived transfer only"},
        {"experiment": "no_transfer_features", "feature_cols": [c for c in all_cols if c not in transfer_family and c not in raw_transfer_set and c not in funnel_transfer_dupes], "note": "remove transfer raw and derived"},
    ]


def add_context_funnel_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    fresh_ctx = (out.get("is_fresh_embryo", 0).eq(1)) & (out.get("reason_current_treatment", 0).eq(1)) & (out.get("embryo_transferred_flag", 0).eq(1))
    frozen_ctx = (out.get("is_frozen_embryo", 0).eq(1)) | (out.get("has_embryo_thaw", 0).eq(1))
    for src, dst in [
        ("embryo_creation_rate", "fresh_current_transfer_embryo_creation_rate"),
        ("transfer_per_embryo_rate", "fresh_current_transfer_transfer_per_embryo_rate"),
        ("storage_per_embryo_rate", "fresh_current_transfer_storage_per_embryo_rate"),
    ]:
        out[dst] = out[src].where(fresh_ctx) if src in out else np.nan
    out["frozen_thawed_embryo_transfer_proxy"] = out["thawed_embryo_transfer_proxy"].where(frozen_ctx) if "thawed_embryo_transfer_proxy" in out else np.nan
    out["frozen_embryo_thawed_count_log1p"] = out["embryo_thawed_count_log1p"].where(frozen_ctx) if "embryo_thawed_count_log1p" in out else np.nan
    out["frozen_transfer_day"] = out["transfer_day_raw"].where(frozen_ctx) if "transfer_day_raw" in out else np.nan
    return out


def funnel_experiments(families: dict[str, list[str]]) -> list[dict[str, Any]]:
    all_fams = list(families.keys())
    no_funnel = [f for f in all_fams if f not in {"funnel_count_bin_features", "funnel_ratio_features"}]
    branch_ratio_flags = ["transfer_gt_created_flag", "fresh_current_funnel_valid_flag", "frozen_funnel_flag", "icsi_funnel_flag"]
    global_ratios = [c for c in families.get("funnel_ratio_features", []) if c not in branch_ratio_flags]
    c5 = ["fresh_current_transfer_embryo_creation_rate", "fresh_current_transfer_transfer_per_embryo_rate", "fresh_current_transfer_storage_per_embryo_rate"]
    c6 = ["frozen_thawed_embryo_transfer_proxy", "frozen_embryo_thawed_count_log1p", "frozen_transfer_day"]
    return [
        {"experiment": "all_no_funnel", "feature_cols": families_to_columns(families, no_funnel)},
        {"experiment": "funnel_counts_only_added", "feature_cols": families_to_columns(families, no_funnel + ["funnel_count_bin_features"])},
        {"experiment": "funnel_ratios_only_added", "feature_cols": families_to_columns(families, no_funnel + ["funnel_ratio_features"])},
        {"experiment": "funnel_counts_and_ratios", "feature_cols": families_to_columns(families, no_funnel + ["funnel_count_bin_features", "funnel_ratio_features"])},
        {"experiment": "no_global_ratio_keep_branch_flags", "feature_cols": [c for c in families_to_columns(families, all_fams) if c not in global_ratios]},
        {"experiment": "fresh_context_only_funnel", "feature_cols": families_to_columns(families, no_funnel) + c5},
        {"experiment": "frozen_context_only_funnel", "feature_cols": families_to_columns(families, no_funnel) + c6},
    ]


def subgroup_auc_for_oof(X: pd.DataFrame, oof: pd.DataFrame) -> dict[str, float]:
    frame = X[["embryo_transferred_flag", "is_fresh_embryo", "is_frozen_embryo", "reason_current_treatment", "is_donor_egg"]].copy()
    frame["row_index"] = np.arange(len(frame))
    m = oof.merge(frame, on="row_index", how="left")
    out = {}
    for col in ["embryo_transferred_flag", "is_fresh_embryo", "is_frozen_embryo", "reason_current_treatment", "is_donor_egg"]:
        sub = m[m[col].eq(1)]
        out[f"auc_{col}=1"] = compute_auc(sub["y_true"], sub["oof_pred"]) if len(sub) and sub["y_true"].nunique() > 1 else np.nan
    return out


def summarize_subgroup_errors(X: pd.DataFrame, raw: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = X.copy()
    frame["row_index"] = np.arange(len(frame))
    frame["ID"] = raw[ID_COLUMN].astype(str).to_numpy()
    m = oof.merge(frame.drop(columns=["ID"]), on="row_index", how="left")

    specs: list[tuple[str, str, pd.Series]] = [("overall", "all", pd.Series(True, index=m.index))]
    for raw_col, name in [("시술 유형", "treatment_type"), ("age_group_raw", "age_group"), ("egg_source_raw", "egg_source"), ("sperm_source_raw", "sperm_source")]:
        if raw_col in m:
            for value in sorted(m[raw_col].dropna().astype(str).unique()):
                specs.append((name, value, m[raw_col].astype(str).eq(value)))
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
        if col in m:
            specs.append((col, "1", m[col].eq(1)))

    rows = []
    for name, value, mask in specs:
        sub = m[mask.fillna(False)]
        if sub.empty:
            continue
        y_true = sub["y_true"]
        pred = sub["oof_pred"]
        pos_pred = pred[y_true.eq(1)].mean() if y_true.eq(1).any() else np.nan
        neg_pred = pred[y_true.eq(0)].mean() if y_true.eq(0).any() else np.nan
        rows.append(
            {
                "subgroup": name,
                "value": value,
                "n": len(sub),
                "positive_rate": y_true.mean(),
                "pred_mean": pred.mean(),
                "pred_std": pred.std(),
                "auc": compute_auc(y_true, pred) if y_true.nunique() > 1 else np.nan,
                "brier_score": brier_score_loss(y_true, pred) if y_true.nunique() > 1 else np.nan,
                "calibration_gap": pred.mean() - y_true.mean(),
                "mean_pred_positive": pos_pred,
                "mean_pred_negative": neg_pred,
                "separation": pos_pred - neg_pred if pd.notna(pos_pred) and pd.notna(neg_pred) else np.nan,
                "note": "" if y_true.nunique() > 1 else "only one class present",
            }
        )
    diag = pd.DataFrame(rows)
    summary = diag.sort_values(["auc", "n"], ascending=[True, False], na_position="last").head(30)
    return diag, summary


def make_recommendations(family_results: pd.DataFrame, transfer_results: pd.DataFrame, funnel_results: pd.DataFrame, specialist_results: pd.DataFrame) -> pd.DataFrame:
    raw_auc = family_results.loc[family_results["experiment"].eq("raw_only"), "mean_auc"].iloc[0]
    all_auc = family_results.loc[family_results["experiment"].eq("all_features"), "mean_auc"].iloc[0]

    def delta(exp: str) -> float:
        vals = family_results.loc[family_results["experiment"].eq(exp), "mean_auc"]
        return float(vals.iloc[0] - raw_auc) if len(vals) else np.nan

    rows = []
    decisions = {
        "branch_flags": ("keep" if delta("raw_plus_branch") > 0 else "needs_full_training_test", delta("raw_plus_branch")),
        "treatment_tokens": ("keep" if delta("raw_plus_treatment_tokens") > 0 else "keep_but_simplify", delta("raw_plus_treatment_tokens")),
        "embryo_reason_features": ("keep" if delta("raw_plus_embryo_reason") > 0 else "keep_but_simplify", delta("raw_plus_embryo_reason")),
        "age_source_interactions": ("keep" if delta("raw_plus_age_source") > 0 else "needs_full_training_test", delta("raw_plus_age_source")),
        "transfer_features": ("keep" if delta("raw_plus_transfer") > 0 else "keep_but_simplify", delta("raw_plus_transfer")),
        "funnel_count_bin_features": ("keep" if delta("raw_plus_funnel_counts") > 0 else "keep_but_simplify", delta("raw_plus_funnel_counts")),
        "funnel_ratio_features": ("keep" if delta("raw_plus_funnel_ratios") > 0 else "needs_branch_specific_test", delta("raw_plus_funnel_ratios")),
        "history_features": ("keep" if delta("raw_plus_history") > 0 else "needs_full_training_test", delta("raw_plus_history")),
        "cause_features": ("keep" if delta("raw_plus_cause") > 0 else "needs_full_training_test", delta("raw_plus_cause")),
    }
    for item, (decision, d) in decisions.items():
        rows.append(
            {
                "recommendation_level": "family",
                "feature_family_or_feature": item,
                "decision": decision,
                "reason": f"mean AUC delta vs raw_only={d:.6f}; all_features mean AUC={all_auc:.6f}",
                "evidence_table": "feature_family_ablation_v2.csv",
                "risk_or_caution": "fast diagnostic mode; confirm under full training",
                "next_action": "carry into next full training if stable",
            }
        )
    b2 = transfer_results.loc[transfer_results["experiment"].eq("transfer_minimal"), "mean_auc"].iloc[0]
    b3 = transfer_results.loc[transfer_results["experiment"].eq("transfer_expanded"), "mean_auc"].iloc[0]
    rows.append(
        {
            "recommendation_level": "feature_set",
            "feature_family_or_feature": "transfer_minimal vs transfer_expanded",
            "decision": "keep_but_simplify" if abs(b3 - b2) < 0.0005 else "keep",
            "reason": f"transfer_minimal={b2:.6f}, transfer_expanded={b3:.6f}",
            "evidence_table": "transfer_feature_ablation.csv",
            "risk_or_caution": "transfer features are highly correlated",
            "next_action": "prefer simpler set if full training confirms parity",
        }
    )
    best_spec_delta = specialist_results["delta_vs_global_subset_auc"].max(skipna=True)
    rows.append(
        {
            "recommendation_level": "modeling",
            "feature_family_or_feature": "specialist model 필요성",
            "decision": "needs_branch_specific_test" if best_spec_delta > 0.002 else "needs_full_training_test",
            "reason": f"best specialist delta vs global subset AUC={best_spec_delta:.6f}",
            "evidence_table": "specialist_model_diagnostics.csv",
            "risk_or_caution": "specialists are not used for submission in this stage",
            "next_action": "test branch blending after full global baseline",
        }
    )
    return pd.DataFrame(rows)


def build_report(tables: dict[str, pd.DataFrame]) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def t(name: str, n: int = 25) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    best = tables["feature_family_ablation_v2"].sort_values("mean_auc", ascending=False).iloc[0]
    html_parts = [
        f"<style>{css}</style>",
        "<h1>Feature Diagnostics v2 Report</h1>",
        f"<p class='note'>{html.escape(LEAKAGE_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best diagnostic experiment: {html.escape(str(best['experiment']))}, mean AUC {best['mean_auc']:.6f}. This is fast diagnostics, not full training.</p>",
        "<h2>Leakage-safe Scope</h2><p>Only train.csv was used. No submission or test diagnostics were created.</p>",
        "<h2>Why this is still Feature Diagnostics, not Full Training</h2><p>Iterations and folds were reduced to compare feature sets consistently.</p>",
        "<h2>Feature Family Ablation Results</h2>", t("feature_family_ablation_v2", 40),
        "<h2>Transfer Feature Redundancy Analysis</h2>", t("transfer_feature_ablation", 20),
        "<h2>Branch-aware Funnel Analysis</h2>", t("funnel_branch_ablation", 20),
        "<h2>Specialist Model Diagnostics</h2>", t("specialist_model_diagnostics", 20),
        "<h2>Subgroup Error Diagnostics</h2>", t("subgroup_error_summary", 30),
        "<h2>Feature Importance Comparison</h2>", t("feature_importance_by_experiment", 50),
        "<h2>Recommended Feature Set for Next Stage</h2>", t("feature_set_recommendation", 50),
        "<h2>Open Questions</h2><p>Whether transfer simplification and specialist models survive full CatBoost/LightGBM training remains open.</p>",
        "<h2>Next Step Plan</h2><p>Run full training on the recommended feature set, then compare branch-specific blending without using test distribution feedback.</p>",
    ]
    (OUT_DIR / "diagnostics_v2_report.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
        X.index = train[ID_COLUMN].astype(str) if ID_COLUMN in train else X.index.astype(str)
        X = add_context_funnel_features(X)
        families = save_feature_families(OUT_DIR / "feature_list_by_family.json", X.columns.tolist())
        for extra in [
            "fresh_current_transfer_embryo_creation_rate",
            "fresh_current_transfer_transfer_per_embryo_rate",
            "fresh_current_transfer_storage_per_embryo_rate",
            "frozen_thawed_embryo_transfer_proxy",
            "frozen_embryo_thawed_count_log1p",
            "frozen_transfer_day",
        ]:
            families.setdefault("funnel_ratio_features", []).append(extra)
        lookup = build_family_lookup(families)
        save_json(OUT_DIR / "diagnostics_v2_config.json", {"catboost": CATBOOST_CONFIG, "cv_splits": CV_SPLITS, "note": FAST_NOTE})

        family_rows, oofs, imps = [], [], []
        raw_auc = all_auc = np.nan
        for spec in family_ablation_experiments(families):
            row, oof, imp = run_cv(X, y, spec["feature_cols"], spec["experiment"], lookup)
            row.update({k: spec[k] for k in ["feature_set_type", "included_families", "excluded_families"]})
            family_rows.append(row)
            if spec["experiment"] in {"raw_only", "all_features", "all_minus_transfer"}:
                oofs.append(oof)
                imps.append(imp)
            completed.append(spec["experiment"])
        family_df = pd.DataFrame(family_rows)
        raw_auc = family_df.loc[family_df["experiment"].eq("raw_only"), "mean_auc"].iloc[0]
        all_auc = family_df.loc[family_df["experiment"].eq("all_features"), "mean_auc"].iloc[0]
        family_df["delta_vs_raw"] = family_df["mean_auc"] - raw_auc
        family_df["delta_vs_all"] = family_df["mean_auc"] - all_auc

        all_cols = all_feature_columns(families)
        transfer_rows = []
        for spec in transfer_experiments(families, all_cols):
            row, oof, imp = run_cv(X, y, [c for c in spec["feature_cols"] if c in X.columns], spec["experiment"], lookup)
            row["diagnostic_note"] = spec["note"]
            transfer_rows.append(row)
            if spec["experiment"] in {"transfer_minimal", "transfer_expanded"}:
                oofs.append(oof)
                imps.append(imp)
            completed.append(spec["experiment"])
        transfer_df = pd.DataFrame(transfer_rows)

        funnel_rows = []
        for spec in funnel_experiments(families):
            cols = [c for c in spec["feature_cols"] if c in X.columns]
            row, oof, imp = run_cv(X, y, cols, spec["experiment"], lookup)
            row.update(subgroup_auc_for_oof(X, oof) if not oof.empty else {})
            funnel_rows.append(row)
            if spec["experiment"] == "funnel_counts_and_ratios":
                oofs.append(oof)
                imps.append(imp)
            completed.append(spec["experiment"])
        funnel_df = pd.DataFrame(funnel_rows)

        all_oof = next(o for o in oofs if not o.empty and o["experiment"].iloc[0] == "all_features")
        specialist_specs = [
            ("global_model", pd.Series(True, index=X.index), "D0 existing all_features global OOF"),
            ("transfer_positive_specialist", X["embryo_transferred_flag"].eq(1), "embryo_transferred_flag=1"),
            ("day5_specialist", X["transfer_day_5"].eq(1), "transfer_day_5=1"),
            ("frozen_specialist", X["is_frozen_embryo"].eq(1), "is_frozen_embryo=1"),
            ("donor_egg_specialist", X["is_donor_egg"].eq(1), "is_donor_egg=1"),
            ("di_specialist", X["is_di"].eq(1), "is_di=1"),
        ]
        spec_rows = []
        for name, mask, note in specialist_specs:
            global_subset = all_oof[all_oof["row_index"].isin(np.flatnonzero(mask.to_numpy()))]
            global_auc = compute_auc(global_subset["y_true"], global_subset["oof_pred"]) if len(global_subset) and global_subset["y_true"].nunique() > 1 else np.nan
            if name == "global_model":
                row = {
                    "specialist": name,
                    "n": len(y),
                    "positive_rate": y.mean(),
                    "fold_auc_list": "",
                    "mean_auc": compute_auc(all_oof["y_true"], all_oof["oof_pred"]),
                    "std_auc": np.nan,
                    "global_model_auc_on_same_subset": global_auc,
                    "delta_vs_global_subset_auc": 0.0,
                    "feature_count": len(all_cols),
                    "note": note,
                }
            elif mask.sum() < 100 or y[mask.to_numpy()].nunique() < 2:
                row = {
                    "specialist": name,
                    "n": int(mask.sum()),
                    "positive_rate": y[mask.to_numpy()].mean() if mask.sum() else np.nan,
                    "fold_auc_list": "",
                    "mean_auc": np.nan,
                    "std_auc": np.nan,
                    "global_model_auc_on_same_subset": global_auc,
                    "delta_vs_global_subset_auc": np.nan,
                    "feature_count": len(all_cols),
                    "note": f"skipped: {note}; insufficient rows or one class",
                }
                skipped.append(name)
            else:
                cv_row, _, _ = run_cv(X, y, all_cols, name, lookup, mask)
                row = {
                    "specialist": name,
                    "n": int(mask.sum()),
                    "positive_rate": y[mask.to_numpy()].mean(),
                    "fold_auc_list": cv_row["fold_auc_list"],
                    "mean_auc": cv_row["mean_auc"],
                    "std_auc": cv_row["std_auc"],
                    "global_model_auc_on_same_subset": global_auc,
                    "delta_vs_global_subset_auc": cv_row["oof_auc"] - global_auc,
                    "feature_count": len(all_cols),
                    "note": note,
                }
                completed.append(name)
            spec_rows.append(row)
        specialist_df = pd.DataFrame(spec_rows)

        best_exp = family_df.sort_values("mean_auc", ascending=False).iloc[0]["experiment"]
        best_oof = next((o for o in oofs if not o.empty and o["experiment"].iloc[0] == best_exp), all_oof)
        subgroup_df, subgroup_summary = summarize_subgroup_errors(X, train, best_oof)
        rec_df = make_recommendations(family_df, transfer_df, funnel_df, specialist_df)
        imp_df = pd.concat(imps, ignore_index=True).drop_duplicates(["experiment", "feature"], keep="first") if imps else pd.DataFrame()
        oof_long = pd.concat(oofs, ignore_index=True)

        tables = {
            "feature_family_ablation_v2": family_df,
            "transfer_feature_ablation": transfer_df,
            "funnel_branch_ablation": funnel_df,
            "specialist_model_diagnostics": specialist_df,
            "subgroup_diagnostics_v2": subgroup_df,
            "subgroup_error_summary": subgroup_summary,
            "feature_set_recommendation": rec_df,
            "feature_importance_by_experiment": imp_df,
            "oof_predictions_by_experiment": oof_long[["ID", "y_true", "experiment", "fold", "oof_pred"]],
        }
        for name, df in tables.items():
            save_table(OUT_DIR / f"{name}.csv", df)
        build_report(tables)

        warnings_list.extend(str(w.message) for w in caught)
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"target_column: {target}",
            f"cv_setting: StratifiedKFold(n_splits={CV_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
            f"catboost_setting: {CATBOOST_CONFIG}",
            f"number_of_experiments: {len(completed) + len(skipped)}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments_with_reason: {','.join(skipped) if skipped else 'none'}",
            f"best_diagnostic_experiment: {best_exp}",
            f"best_diagnostic_mean_auc: {family_df.sort_values('mean_auc', ascending=False).iloc[0]['mean_auc']:.6f}",
            f"key_subgroup_findings: {subgroup_summary.head(5).to_dict(orient='records')}",
            f"leakage_note: {LEAKAGE_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "diagnostics_v2_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
