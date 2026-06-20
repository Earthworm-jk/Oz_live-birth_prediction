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

from src.features.art_features import ID_COLUMN, make_art_features, save_feature_families
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup
from src.models.evidence_guided_feature_audit_v1 import (
    PROHIBITED_NOTE,
    V2_CONFIGS,
    add_evidence_features,
    build_feature_sets,
    run_catboost_cv,
)
from src.models.model_branch_diagnostics_v1 import (
    LGBM_CONFIG,
    load_catboost_oof,
    run_lgbm_cv,
    subgroup_auc_by_model,
)
from src.models.model_utils import compute_auc, get_target_column, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "evidence_model_expansion_v1"

OLD_CB_NAME = "CB_long_all_features_depth6"
OLD_LGBM_NAME = "LGBM_A_all_features"
CBV2_DEPTH7_NAME = "CB_v2_depth7_lr045_l2_6"
CBV2_DEPTH6_NAME = "CB_v2_depth6_lr035_l2_5"
OLD_GLOBAL_BLEND_AUC = 0.740462
CURRENT_BEST_BLEND_AUC = 0.740818
OLD_CATBOOST_AUC = 0.740185
CBV2_DEPTH7_AUC = 0.740650
RANDOM_SEED = 42
N_SPLITS = 5

OLD_LGBM_OOF_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_oof_predictions.csv"
OLD_LGBM_RESULT_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_cv_results.csv"
CBV2_OOF_PATH = PROJECT_ROOT / "outputs" / "evidence_guided_feature_audit_v1" / "catboost_v2_oof_predictions.csv"
CBV2_RESULT_PATH = PROJECT_ROOT / "outputs" / "evidence_guided_feature_audit_v1" / "catboost_v2_cv_results.csv"

LGBM_EXPERIMENTS = [
    ("LGBM_A_all_evidence_features", "all_evidence_features"),
    ("LGBM_A_all_evidence_minus_timecode_interactions", "all_evidence_minus_treatment_time_code_interactions"),
    ("LGBM_A_all_evidence_minus_time_interval", "all_evidence_minus_time_interval_features"),
    ("LGBM_A_all_evidence_minus_egg_age", "all_evidence_minus_egg_biological_age_proxy"),
    ("LGBM_A_all_evidence_minus_set_day5", "all_evidence_minus_set_blastocyst_day5_features"),
]

CATBOOST_EXPERIMENTS = [
    ("CB_v2_all_evidence", "all_evidence_features"),
    ("CB_v2_all_evidence_minus_timecode_interactions", "all_evidence_minus_treatment_time_code_interactions"),
    ("CB_v2_all_evidence_minus_time_interval", "all_evidence_minus_time_interval_features"),
    ("CB_v2_all_evidence_minus_egg_age", "all_evidence_minus_egg_biological_age_proxy"),
    ("CB_v2_all_evidence_minus_set_day5", "all_evidence_minus_set_blastocyst_day5_features"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence Model Expansion v1")
    parser.add_argument("--stage", choices=["all", "lgbm", "minus", "blend"], default="all")
    parser.add_argument("--quick", action="store_true", help="Use reduced estimators/iterations for smoke testing.")
    parser.add_argument("--priority-only", action="store_true", help="Run only priority LGBM 3 and CatBoost minus 3 experiments.")
    return parser.parse_args()


def load_data_and_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    train = pd.read_csv(TRAIN_PATH)
    target = get_target_column(train)
    y = train[target].astype(int)
    X_base = make_art_features(train)
    X_base.index = train[ID_COLUMN].astype(str)
    X, feature_list, skipped = add_evidence_features(X_base, train)
    if not skipped.empty:
        raise ValueError(f"Evidence feature creation skipped features: {skipped.to_dict(orient='records')}")
    base_families = save_feature_families(OUT_DIR / "feature_families_base.json", X_base.columns.tolist())
    base_cols = all_cols_from_families(base_families, X_base)
    feature_sets = build_feature_sets(base_cols, feature_list)
    families = {k: list(v) for k, v in base_families.items()}
    for fam, sub in feature_list.groupby("feature_family"):
        families[f"evidence_{fam}"] = sub["feature_name"].tolist()
    lookup = family_lookup(families)
    save_table(OUT_DIR / "evidence_feature_list.csv", feature_list)
    return train, y, X, feature_sets, lookup


def load_old_lgbm() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OLD_LGBM_OOF_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    oof = pd.read_csv(OLD_LGBM_OOF_PATH)
    oof = oof[oof["model_name"].eq(OLD_LGBM_NAME)].copy()
    if oof.empty:
        return pd.DataFrame(), pd.DataFrame()
    oof = oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]]
    row = {
        "model_name": OLD_LGBM_NAME,
        "feature_set": "all_features_for_lgbm",
        "n_features": np.nan,
        "n_categorical": np.nan,
        "fold_auc_list": "",
        "mean_auc": np.nan,
        "std_auc": np.nan,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": "",
        "delta_vs_lgbm_a_all_features": 0.0,
        "delta_vs_old_catboost": compute_auc(oof["y_true"], oof["oof_pred"]) - OLD_CATBOOST_AUC,
        "delta_vs_old_global_blend": compute_auc(oof["y_true"], oof["oof_pred"]) - OLD_GLOBAL_BLEND_AUC,
        "delta_vs_cbv2_depth7": compute_auc(oof["y_true"], oof["oof_pred"]) - CBV2_DEPTH7_AUC,
        "training_time_sec": 0.0,
        "note": f"Loaded from {OLD_LGBM_OOF_PATH}",
    }
    if OLD_LGBM_RESULT_PATH.exists():
        result = pd.read_csv(OLD_LGBM_RESULT_PATH)
        result = result[result["model_name"].eq(OLD_LGBM_NAME)]
        if not result.empty:
            r = result.iloc[0]
            row.update(
                {
                    "n_features": r.get("feature_count", np.nan),
                    "n_categorical": r.get("categorical_feature_count", np.nan),
                    "fold_auc_list": r.get("fold_auc_list", ""),
                    "mean_auc": r.get("mean_auc", np.nan),
                    "std_auc": r.get("std_auc", np.nan),
                    "best_iteration_list": r.get("best_iteration_list", ""),
                }
            )
    return oof, pd.DataFrame([row])


def load_cbv2_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    oofs = []
    rows = []
    if CBV2_OOF_PATH.exists():
        oof = pd.read_csv(CBV2_OOF_PATH)
        for name in [CBV2_DEPTH7_NAME, CBV2_DEPTH6_NAME]:
            sub = oof[oof["model_name"].eq(name)].copy()
            if not sub.empty:
                oofs.append(sub[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]])
    if CBV2_RESULT_PATH.exists():
        res = pd.read_csv(CBV2_RESULT_PATH)
        rows.append(res[res["model_name"].isin([CBV2_DEPTH7_NAME, CBV2_DEPTH6_NAME])].copy())
    return (
        pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame(),
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
    )


def run_lgbm_stage(args: argparse.Namespace, train: pd.DataFrame, y: pd.Series, X: pd.DataFrame, feature_sets: dict[str, list[str]], lookup: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_oof, old_result = load_old_lgbm()
    rows = [old_result] if not old_result.empty else []
    oofs = [old_oof] if not old_oof.empty else []
    imps = []
    config = dict(LGBM_CONFIG)
    if args.quick:
        config.update({"n_estimators": 350})
    experiments = LGBM_EXPERIMENTS[:3] if args.priority_only else LGBM_EXPERIMENTS
    baseline_auc = compute_auc(old_oof["y_true"], old_oof["oof_pred"]) if not old_oof.empty else np.nan
    for model_name, fs in experiments:
        print(f"[evidence_expansion] LGBM start: {model_name}", flush=True)
        row, oof, imp = run_lgbm_cv(X, y, feature_sets[fs], model_name, fs, config, 3 if args.quick else N_SPLITS, lookup)
        row = {
            "model_name": row["model_name"],
            "feature_set": row["feature_set"],
            "n_features": row["feature_count"],
            "n_categorical": row["categorical_feature_count"],
            "fold_auc_list": row["fold_auc_list"],
            "mean_auc": row["mean_auc"],
            "std_auc": row["std_auc"],
            "oof_auc": row["oof_auc"],
            "best_iteration_list": row["best_iteration_list"],
            "delta_vs_lgbm_a_all_features": row["oof_auc"] - baseline_auc if pd.notna(baseline_auc) else np.nan,
            "delta_vs_old_catboost": row["oof_auc"] - OLD_CATBOOST_AUC,
            "delta_vs_old_global_blend": row["oof_auc"] - OLD_GLOBAL_BLEND_AUC,
            "delta_vs_cbv2_depth7": row["oof_auc"] - CBV2_DEPTH7_AUC,
            "training_time_sec": row["training_time_sec"],
            "note": "LGBM evidence CV; fold-local categorical handling",
        }
        rows.append(pd.DataFrame([row]))
        oofs.append(oof[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]])
        if not imp.empty:
            imps.append(imp)
        print(f"[evidence_expansion] LGBM done: {model_name} oof_auc={row['oof_auc']:.6f}", flush=True)
    result = pd.concat(rows, ignore_index=True).sort_values("oof_auc", ascending=False) if rows else pd.DataFrame()
    oof_all = pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame()
    imp_all = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
    save_table(OUT_DIR / "lgbm_evidence_cv_results.csv", result)
    save_table(OUT_DIR / "lgbm_evidence_oof_predictions.csv", oof_all)
    save_table(OUT_DIR / "lgbm_evidence_feature_importance.csv", imp_all)
    return result, oof_all, imp_all


def run_catboost_minus_stage(args: argparse.Namespace, y: pd.Series, X: pd.DataFrame, feature_sets: dict[str, list[str]], lookup: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = dict(V2_CONFIGS[CBV2_DEPTH7_NAME])
    if args.quick:
        config.update({"iterations": 180, "early_stopping_rounds": 50})
    experiments = CATBOOST_EXPERIMENTS[:3] if args.priority_only else CATBOOST_EXPERIMENTS
    rows = []
    oofs = []
    imps = []
    all_evidence_auc = np.nan
    for model_name, fs in experiments:
        print(f"[evidence_expansion] CatBoost minus start: {model_name}", flush=True)
        row, oof, imp = run_catboost_cv(X, y, feature_sets[fs], model_name, fs, config, 3 if args.quick else N_SPLITS, lookup)
        if fs == "all_evidence_features":
            all_evidence_auc = row["oof_auc"]
        row["delta_vs_all_evidence"] = row["oof_auc"] - all_evidence_auc if pd.notna(all_evidence_auc) else np.nan
        row["delta_vs_old_catboost"] = row["oof_auc"] - OLD_CATBOOST_AUC
        row["delta_vs_old_global_blend"] = row["oof_auc"] - OLD_GLOBAL_BLEND_AUC
        row["delta_vs_current_best_blend_0_740818"] = row["oof_auc"] - CURRENT_BEST_BLEND_AUC
        rows.append(row)
        oofs.append(oof)
        imps.append(imp)
        print(f"[evidence_expansion] CatBoost minus done: {model_name} oof_auc={row['oof_auc']:.6f}", flush=True)
    result = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    oof_all = pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame()
    imp_all = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
    save_table(OUT_DIR / "catboost_minus_family_cv_results.csv", result)
    save_table(OUT_DIR / "catboost_minus_family_oof_predictions.csv", oof_all)
    save_table(OUT_DIR / "catboost_minus_family_feature_importance.csv", imp_all)
    return result, oof_all, imp_all


def prediction_matrix(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.pivot_table(index="ID", columns="model_name", values="oof_pred")


def correlation_table(oof: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    rows = []
    for i, a in enumerate(mat.columns):
        for b in mat.columns[i + 1 :]:
            pair = mat[[a, b]].dropna()
            p = pair[a].corr(pair[b]) if len(pair) else np.nan
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n": len(pair),
                    "pearson": p,
                    "spearman": spearmanr(pair[a], pair[b]).correlation if len(pair) else np.nan,
                    "interpretation": corr_note(p),
                }
            )
    return pd.DataFrame(rows)


def corr_note(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= 0.995:
        return "near identical"
    if value >= 0.990:
        return "weak diversity"
    if value >= 0.970:
        return "possible ensemble value"
    return "large diversity; verify performance"


def make_blends(oof: pd.DataFrame) -> pd.DataFrame:
    mat = prediction_matrix(oof)
    y = oof.drop_duplicates("ID").set_index("ID")["y_true"].loc[mat.index]
    rows = []

    def add(names: list[str], weights: list[float], label: str) -> None:
        if not all(n in mat.columns for n in names):
            return
        pair = mat[names].dropna()
        pred = sum(w * pair[n] for w, n in zip(weights, names))
        auc = compute_auc(y.loc[pair.index], pred)
        rows.append(
            {
                "blend_name": label,
                "base_models": ",".join(names),
                "weights": ",".join(str(w) for w in weights),
                "oof_auc": auc,
                "delta_vs_old_catboost": auc - OLD_CATBOOST_AUC,
                "delta_vs_old_global_blend": auc - OLD_GLOBAL_BLEND_AUC,
                "delta_vs_current_best_blend_0_740818": auc - CURRENT_BEST_BLEND_AUC,
                "note": "OOF-only diagnostic blend; no submission generated",
            }
        )

    two_pairs = [
        ([CBV2_DEPTH7_NAME, "LGBM_A_all_evidence_features"], "cbv2_lgbm_evidence"),
        ([CBV2_DEPTH7_NAME, "LGBM_A_all_evidence_minus_timecode_interactions"], "cbv2_lgbm_minus_timecode"),
        (["CB_v2_all_evidence_minus_timecode_interactions", "LGBM_A_all_evidence_features"], "cb_minus_timecode_lgbm_evidence"),
    ]
    for names, prefix in two_pairs:
        for weights in [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]:
            add(names, list(weights), f"{prefix}_{weights[0]}_{weights[1]}")
    three = [
        ([CBV2_DEPTH7_NAME, OLD_CB_NAME, "LGBM_A_all_evidence_features"], "cbv2_oldcb_lgbm_evidence"),
        (["CB_v2_all_evidence_minus_timecode_interactions", OLD_CB_NAME, "LGBM_A_all_evidence_features"], "cb_minus_timecode_oldcb_lgbm_evidence"),
    ]
    for names, prefix in three:
        for weights in [(0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
            add(names, list(weights), f"{prefix}_{'_'.join(str(w) for w in weights)}")
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False) if rows else pd.DataFrame()


def load_all_oof() -> pd.DataFrame:
    frames = []
    cb_oof, _ = load_catboost_oof(pd.read_csv(TRAIN_PATH))
    frames.append(cb_oof)
    old_lgbm_oof, _ = load_old_lgbm()
    if not old_lgbm_oof.empty:
        frames.append(old_lgbm_oof)
    cbv2_oof, _ = load_cbv2_existing()
    if not cbv2_oof.empty:
        frames.append(cbv2_oof)
    for path in [OUT_DIR / "lgbm_evidence_oof_predictions.csv", OUT_DIR / "catboost_minus_family_oof_predictions.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    return out[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]]


def branch_focus(subgroup: pd.DataFrame) -> pd.DataFrame:
    branch_map = {
        "overall": ("overall", "all"),
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
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        old = sub[sub["model_name"].eq(OLD_CB_NAME)]
        cbv2 = sub[sub["model_name"].eq(CBV2_DEPTH7_NAME)]
        lgbm_ev = sub[sub["model_name"].eq("LGBM_A_all_evidence_features")]
        minus_tc = sub[sub["model_name"].eq("CB_v2_all_evidence_minus_timecode_interactions")]
        old_auc = old["auc"].iloc[0] if len(old) else np.nan
        rows.append(
            {
                "branch_name": branch,
                "n": int(best["n"]),
                "positive_rate": best["positive_rate"],
                "old_catboost_auc": old_auc,
                "cbv2_depth7_auc": cbv2["auc"].iloc[0] if len(cbv2) else np.nan,
                "lgbm_evidence_auc": lgbm_ev["auc"].iloc[0] if len(lgbm_ev) else np.nan,
                "cb_minus_timecode_auc": minus_tc["auc"].iloc[0] if len(minus_tc) else np.nan,
                "best_model_by_auc": best["model_name"],
                "best_auc": best["auc"],
                "delta_best_vs_old_catboost": best["auc"] - old_auc if pd.notna(old_auc) else np.nan,
                "interpretation": "model expansion improves branch" if pd.notna(old_auc) and best["auc"] > old_auc else "monitor",
            }
        )
    return pd.DataFrame(rows)


def recommendation_table(models: pd.DataFrame, blends: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_rows = []
    for path, name_col in [
        (OUT_DIR / "lgbm_evidence_cv_results.csv", "model_name"),
        (OUT_DIR / "catboost_minus_family_cv_results.csv", "model_name"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                if r[name_col] == OLD_LGBM_NAME:
                    continue
                model_rows.append(
                    {
                        "model_or_blend_name": r[name_col],
                        "oof_auc": float(r["oof_auc"]),
                        "feature_set": r.get("feature_set", ""),
                    }
                )
    for row in model_rows:
        rows.append(make_reco_row(row["model_or_blend_name"], row["oof_auc"], row["feature_set"]))
    for _, r in blends.iterrows():
        rows.append(make_reco_row(r["blend_name"], float(r["oof_auc"]), r["base_models"]))
    out = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    if not out.empty:
        out["rank"] = np.arange(1, len(out) + 1)
        out = out[["rank", "model_or_blend_name", "oof_auc", "delta_vs_old_catboost", "delta_vs_old_global_blend", "delta_vs_current_best_blend_0_740818", "feature_set", "risk_level", "recommend_next_submission", "reason"]]
    return out


def make_reco_row(name: str, auc: float, feature_set: str) -> dict[str, Any]:
    risk = "medium"
    if "minus_timecode" in name:
        risk = "private_stability_candidate"
    if auc >= CURRENT_BEST_BLEND_AUC + 0.00020:
        rec = "strong_candidate"
        reason = "OOF exceeds current best blend by >= 0.00020"
    elif auc >= CURRENT_BEST_BLEND_AUC:
        rec = "candidate"
        reason = "OOF matches or exceeds current best blend"
    elif auc >= CBV2_DEPTH7_AUC:
        rec = "hold"
        reason = "OOF is below current best blend but above CB_v2 depth7 single"
    else:
        rec = "reject"
        reason = "OOF below CB_v2 depth7 single"
    return {
        "rank": 0,
        "model_or_blend_name": name,
        "oof_auc": auc,
        "delta_vs_old_catboost": auc - OLD_CATBOOST_AUC,
        "delta_vs_old_global_blend": auc - OLD_GLOBAL_BLEND_AUC,
        "delta_vs_current_best_blend_0_740818": auc - CURRENT_BEST_BLEND_AUC,
        "feature_set": feature_set,
        "risk_level": risk,
        "recommend_next_submission": rec,
        "reason": reason,
    }


def run_blend_stage(train: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = load_all_oof()
    blends = make_blends(oof)
    corr = correlation_table(oof)
    blend_oofs = []
    if not blends.empty:
        mat = prediction_matrix(oof)
        y = oof.drop_duplicates("ID").set_index("ID")["y_true"]
        for _, r in blends.iterrows():
            names = r["base_models"].split(",")
            weights = [float(x) for x in r["weights"].split(",")]
            pair = mat[names].dropna()
            pred = sum(w * pair[n] for w, n in zip(weights, names))
            blend_oofs.append(pd.DataFrame({"ID": pair.index, "y_true": y.loc[pair.index].to_numpy(), "fold": 0, "model_name": "blend_" + r["blend_name"], "feature_set": "diagnostic_blend", "oof_pred": pred.to_numpy()}))
    subgroup_oof = pd.concat([oof] + blend_oofs, ignore_index=True) if blend_oofs else oof
    subgroup = subgroup_auc_by_model(X, train, subgroup_oof, pd.DataFrame())
    branch = branch_focus(subgroup)
    reco = recommendation_table(pd.DataFrame(), blends)
    save_table(OUT_DIR / "diagnostic_blend_results.csv", blends)
    save_table(OUT_DIR / "model_oof_correlation.csv", corr)
    save_table(OUT_DIR / "subgroup_auc_results.csv", subgroup)
    save_table(OUT_DIR / "branch_focus_diagnostics.csv", branch)
    save_table(OUT_DIR / "candidate_recommendation.csv", reco)
    return blends, corr, branch, reco


def build_report() -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def read(name: str) -> pd.DataFrame:
        path = OUT_DIR / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def table(name: str, n: int = 30) -> str:
        return read(name).head(n).to_html(index=False, escape=True, border=0)

    lgbm = read("lgbm_evidence_cv_results.csv")
    cb = read("catboost_minus_family_cv_results.csv")
    blends = read("diagnostic_blend_results.csv")
    best_lgbm = lgbm.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in lgbm.columns else pd.DataFrame()
    best_cb = cb.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in cb.columns else pd.DataFrame()
    best_blend = blends.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in blends.columns else pd.DataFrame()
    summary = (
        f"Best LGBM: {best_lgbm.iloc[0]['model_name']} {float(best_lgbm.iloc[0]['oof_auc']):.6f}. " if len(best_lgbm) else ""
    )
    summary += (
        f"Best CatBoost minus-family: {best_cb.iloc[0]['model_name']} {float(best_cb.iloc[0]['oof_auc']):.6f}. " if len(best_cb) else ""
    )
    summary += (
        f"Best diagnostic blend: {best_blend.iloc[0]['blend_name']} {float(best_blend.iloc[0]['oof_auc']):.6f}." if len(best_blend) else ""
    )
    parts = [
        f"<style>{css}</style>",
        "<h1>Evidence Model Expansion v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>{html.escape(summary)}</p>",
        "<h2>Prohibited Methods and Leakage Safety</h2>",
        "<p>No pseudo-labeling, target encoding, rank normalization, MICE imputation, train+test concat, test-wide post-processing, public-LB-based tuning, or automatic submission was used.</p>",
        "<h2>Why LGBM Evidence Expansion Was Tested</h2>",
        "<p>CatBoost benefited from all_evidence_features; this stage checks whether LightGBM also benefits and whether timecode interactions are necessary or risky.</p>",
        "<h2>LGBM Evidence CV Results</h2>",
        table("lgbm_evidence_cv_results.csv", 20),
        "<h2>CatBoost Minus-Family CV Results</h2>",
        table("catboost_minus_family_cv_results.csv", 20),
        "<h2>Diagnostic Blend Results</h2>",
        table("diagnostic_blend_results.csv", 30),
        "<h2>OOF Correlation</h2>",
        table("model_oof_correlation.csv", 40),
        "<h2>Subgroup AUC</h2>",
        table("subgroup_auc_results.csv", 80),
        "<h2>Branch Focus Diagnostics</h2>",
        table("branch_focus_diagnostics.csv", 30),
        "<h2>Feature Importance</h2>",
        table("lgbm_evidence_feature_importance.csv", 40) + table("catboost_minus_family_feature_importance.csv", 40),
        "<h2>Candidate Recommendation</h2>",
        table("candidate_recommendation.csv", 20),
        "<h2>Interpretation</h2>",
        "<p>Only OOF diagnostics are produced here; submission creation is deferred to a separate stage.</p>",
        "<h2>Next Step</h2>",
        "<p>If a candidate exceeds the current best blend by the stated threshold, generate a dedicated submission candidate in a separate script.</p>",
    ]
    (OUT_DIR / "evidence_model_expansion_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        train, y, X, feature_sets, lookup = load_data_and_features()

        lgbm_results = pd.DataFrame()
        cb_results = pd.DataFrame()
        blends = pd.DataFrame()
        if args.stage in {"all", "lgbm"}:
            lgbm_results, _, _ = run_lgbm_stage(args, train, y, X, feature_sets, lookup)
            completed.append("lgbm")
        else:
            skipped.append("lgbm: skipped by stage")
        if args.stage in {"all", "minus"}:
            cb_results, _, _ = run_catboost_minus_stage(args, y, X, feature_sets, lookup)
            completed.append("catboost_minus")
        else:
            skipped.append("catboost_minus: skipped by stage")
        if args.stage in {"all", "blend"}:
            blends, _, _, _ = run_blend_stage(train, X)
            completed.append("blend")
        else:
            skipped.append("blend: skipped by stage")

        save_json(
            OUT_DIR / "model_config.json",
            {
                "cv": f"StratifiedKFold(n_splits={3 if args.quick else N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
                "stage": args.stage,
                "quick": args.quick,
                "priority_only": args.priority_only,
                "lgbm_config": {**LGBM_CONFIG, **({"n_estimators": 350} if args.quick else {})},
                "catboost_config": {**V2_CONFIGS[CBV2_DEPTH7_NAME], **({"iterations": 180, "early_stopping_rounds": 50} if args.quick else {})},
                "prohibited_methods": PROHIBITED_NOTE,
            },
        )
        build_report()

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import lightgbm
        import sklearn

        if lgbm_results.empty and (OUT_DIR / "lgbm_evidence_cv_results.csv").exists():
            lgbm_results = pd.read_csv(OUT_DIR / "lgbm_evidence_cv_results.csv")
        if cb_results.empty and (OUT_DIR / "catboost_minus_family_cv_results.csv").exists():
            cb_results = pd.read_csv(OUT_DIR / "catboost_minus_family_cv_results.csv")
        if blends.empty and (OUT_DIR / "diagnostic_blend_results.csv").exists():
            blends = pd.read_csv(OUT_DIR / "diagnostic_blend_results.csv")
        reco = pd.read_csv(OUT_DIR / "candidate_recommendation.csv") if (OUT_DIR / "candidate_recommendation.csv").exists() else pd.DataFrame()
        best_lgbm = lgbm_results.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in lgbm_results.columns else pd.DataFrame()
        best_cb = cb_results.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in cb_results.columns else pd.DataFrame()
        best_blend = blends.sort_values("oof_auc", ascending=False).head(1) if "oof_auc" in blends.columns else pd.DataFrame()
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"sklearn_version: {sklearn.__version__}",
            f"catboost_version: {catboost.__version__}",
            f"lightgbm_version: {lightgbm.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"target_column: {get_target_column(train)}",
            f"cv_setting: StratifiedKFold(n_splits={3 if args.quick else N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})",
            f"prohibited_methods_note: {PROHIBITED_NOTE}",
            f"loaded_previous_oof_files: old_lgbm={OLD_LGBM_OOF_PATH}; cbv2={CBV2_OOF_PATH}",
            f"completed_lgbm_experiments: {','.join(lgbm_results['model_name'].astype(str).tolist()) if not lgbm_results.empty else 'none'}",
            f"completed_catboost_experiments: {','.join(cb_results['model_name'].astype(str).tolist()) if not cb_results.empty else 'none'}",
            f"completed_stages: {','.join(completed)}",
            f"skipped_experiments_and_reasons: {','.join(skipped) if skipped else 'none'}",
            f"best_lgbm_model: {best_lgbm.iloc[0]['model_name'] if len(best_lgbm) else 'none'}",
            f"best_lgbm_oof_auc: {best_lgbm.iloc[0]['oof_auc'] if len(best_lgbm) else np.nan}",
            f"best_catboost_minus_family_model: {best_cb.iloc[0]['model_name'] if len(best_cb) else 'none'}",
            f"best_catboost_minus_family_oof_auc: {best_cb.iloc[0]['oof_auc'] if len(best_cb) else np.nan}",
            f"best_diagnostic_blend: {best_blend.iloc[0]['blend_name'] if len(best_blend) else 'none'}",
            f"best_oof_auc: {best_blend.iloc[0]['oof_auc'] if len(best_blend) else (best_cb.iloc[0]['oof_auc'] if len(best_cb) else (best_lgbm.iloc[0]['oof_auc'] if len(best_lgbm) else np.nan))}",
            f"recommendation: {reco.head(5).to_dict(orient='records') if not reco.empty else []}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "evidence_model_expansion_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
