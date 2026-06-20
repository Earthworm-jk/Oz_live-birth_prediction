from __future__ import annotations

import argparse
import html
import platform
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from src.features.art_features import ID_COLUMN, make_art_features
from src.models.model_branch_diagnostics_v1 import subgroup_specs
from src.models.model_utils import compute_auc, get_target_column, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"

CB_OOF_PATH = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "long_oof_predictions.csv"
BRANCH_OOF_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_oof_predictions.csv"
ZOO_OOF_PATH = PROJECT_ROOT / "outputs" / "model_zoo_v1" / "model_zoo_oof_predictions.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "branch_soft_routing_v1"
OPTIONAL_SUB_DIR = OUT_DIR / "optional_submissions"
RANDOM_SEED = 42
CB_NAME = "CB_long_all_features_depth6"
LGBM_NAME = "LGBM_A_all_features"
ET_NAME = "ET_structured_numeric"
XGB_A_NAME = "XGB_A_structured_numeric"
XGB_B_NAME = "XGB_B_ohe_light_categorical"
GLOBAL_BLEND_NAME = "global_cb_lgbm_60_40"
PREVIOUS_GLOBAL_BLEND_AUC = 0.740462
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, target encoding, train+test concat, test-wide post-processing, "
    "public-LB-based routing, and any use of test data for training or routing decisions were not used."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Branch Soft Routing v1")
    parser.add_argument("--make-optional-submissions", action="store_true", help="Only create candidates if OOF criteria are met. Currently diagnostics-first and disabled unless criteria pass.")
    return parser.parse_args()


def load_oof() -> tuple[pd.DataFrame, dict[str, str]]:
    loaded_paths = {
        CB_NAME: str(CB_OOF_PATH),
        LGBM_NAME: str(BRANCH_OOF_PATH),
        ET_NAME: str(ZOO_OOF_PATH),
    }
    cb = pd.read_csv(CB_OOF_PATH)
    cb = cb[cb["experiment"].eq("all_features_long_depth6")].copy() if "experiment" in cb.columns else cb.copy()
    cb = cb[["ID", "y_true", "fold", "oof_pred"]].rename(columns={"oof_pred": CB_NAME, "fold": f"{CB_NAME}__fold"})
    cb["ID"] = cb["ID"].astype(str)

    branch = pd.read_csv(BRANCH_OOF_PATH)
    lgbm = branch[branch["model_name"].eq(LGBM_NAME)][["ID", "fold", "oof_pred"]].copy()
    lgbm = lgbm.rename(columns={"oof_pred": LGBM_NAME, "fold": f"{LGBM_NAME}__fold"})
    lgbm["ID"] = lgbm["ID"].astype(str)

    zoo = pd.read_csv(ZOO_OOF_PATH)
    keep_models = [ET_NAME, XGB_A_NAME, XGB_B_NAME]
    zoo_parts = []
    for model in keep_models:
        sub = zoo[zoo["model_name"].eq(model)]
        if sub.empty:
            continue
        loaded_paths[model] = str(ZOO_OOF_PATH)
        zoo_parts.append(sub[["ID", "fold", "model_name", "oof_pred"]].copy())

    frame = cb.merge(lgbm, on="ID", how="inner", validate="one_to_one")
    for part in zoo_parts:
        model = part["model_name"].iloc[0]
        wide = part[["ID", "fold", "oof_pred"]].rename(columns={"fold": f"{model}__fold", "oof_pred": model})
        wide["ID"] = wide["ID"].astype(str)
        frame = frame.merge(wide, on="ID", how="left", validate="one_to_one")
    return frame, loaded_paths


def prediction_oof_long(preds: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    fold = preds[f"{CB_NAME}__fold"].to_numpy()
    for col in [c for c in preds.columns if c.startswith("global_") or c.startswith("route_") or c.startswith("router_")]:
        rows.append(
            pd.DataFrame(
                {
                    "ID": preds["ID"].astype(str).to_numpy(),
                    "y_true": y.to_numpy(),
                    "fold": fold,
                    "routing_name": col,
                    "oof_pred": preds[col].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def blend(base: pd.Series, aux: pd.Series, aux_weight: float) -> pd.Series:
    return (1.0 - aux_weight) * base + aux_weight * aux


def apply_route(base: pd.Series, aux: pd.Series, mask: pd.Series, aux_weight: float) -> pd.Series:
    pred = base.copy()
    pred.loc[mask] = blend(base.loc[mask], aux.loc[mask], aux_weight)
    return pred


def sequential_router(base: pd.Series, masks: dict[str, pd.Series], aux: dict[str, pd.Series], steps: list[tuple[str, str, float]]) -> pd.Series:
    pred = base.copy()
    for mask_name, aux_name, aux_weight in steps:
        mask = masks[mask_name]
        pred.loc[mask] = blend(pred.loc[mask], aux[aux_name].loc[mask], aux_weight)
    return pred


def build_predictions(oof: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = oof.copy()
    cb = preds[CB_NAME]
    lgbm = preds[LGBM_NAME]
    et = preds[ET_NAME]
    masks = {
        "donor_egg": X["is_donor_egg"].eq(1),
        "no_transfer": X["no_embryo_transfer_flag"].eq(1),
        "storage_only": X["storage_only_flag"].eq(1),
        "frozen": X["is_frozen_embryo"].eq(1),
        "transfer_positive": X["embryo_transferred_flag"].eq(1),
        "day5": X["transfer_day_5"].eq(1),
    }
    aux = {"lgbm": lgbm, "et": et}
    metadata: list[dict[str, Any]] = []

    def add_meta(name: str, routing_type: str, aux_models: str, condition: str, weights: str, affected_mask: pd.Series, risk: str, note: str) -> None:
        metadata.append(
            {
                "routing_name": name,
                "routing_type": routing_type,
                "base_model": CB_NAME,
                "aux_models": aux_models,
                "conditions": condition,
                "weights": weights,
                "affected_n": int(affected_mask.sum()),
                "affected_rate": float(affected_mask.mean()),
                "affected_positive_rate": np.nan,
                "risk_level": risk,
                "note": note,
            }
        )

    preds["global_cb_only"] = cb
    add_meta("global_cb_only", "global_reference", "", "all", "1.0", pd.Series(True, index=preds.index), "low", "CatBoost baseline")
    for aux_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        name = f"global_cb_lgbm_{int((1-aux_w)*100)}_{int(aux_w*100)}"
        preds[name] = blend(cb, lgbm, aux_w)
        add_meta(name, "global_reference", LGBM_NAME, "all", f"{1-aux_w:.2f},{aux_w:.2f}", pd.Series(True, index=preds.index), "low", "Global OOF-only CB+LGBM blend")

    for aux_w in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        name = f"route_donor_lgbm_{int((1-aux_w)*100)}_{int(aux_w*100)}" if aux_w < 1 else "route_donor_lgbm_100"
        preds[name] = apply_route(cb, lgbm, masks["donor_egg"], aux_w)
        add_meta(name, "single_branch", LGBM_NAME, "is_donor_egg == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["donor_egg"], "medium" if aux_w < 1 else "high", "Donor egg LGBM routing; 100% is hard replacement diagnostic")

    for aux_w in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        name = f"route_no_transfer_et_{int((1-aux_w)*100)}_{int(aux_w*100)}" if aux_w < 1 else "route_no_transfer_et_100"
        preds[name] = apply_route(cb, et, masks["no_transfer"], aux_w)
        add_meta(name, "single_branch", ET_NAME, "no_embryo_transfer_flag == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["no_transfer"], "medium" if aux_w <= 0.2 else "high", "No-transfer ET routing; hard/high-weight variants are diagnostic")

    for aux_w, suffix in [(0.02, "98_02"), (0.05, "95_05"), (0.1, "90_10"), (0.2, "80_20"), (0.5, "50_50_diagnostic_only"), (1.0, "100_diagnostic_only")]:
        name = f"route_storage_et_{suffix}"
        preds[name] = apply_route(cb, et, masks["storage_only"], aux_w)
        add_meta(name, "single_branch", ET_NAME, "storage_only_flag == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["storage_only"], "medium" if aux_w <= 0.1 else "high", "Storage-only cautious ET routing; 50/100 diagnostic only")

    for aux_w in [0.1, 0.2, 0.3, 0.4]:
        name = f"route_frozen_lgbm_{int((1-aux_w)*100)}_{int(aux_w*100)}"
        preds[name] = apply_route(cb, lgbm, masks["frozen"], aux_w)
        add_meta(name, "single_branch", LGBM_NAME, "is_frozen_embryo == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["frozen"], "medium", "Frozen low-weight LGBM routing")

    for aux_w in [0.1, 0.2, 0.3, 0.4]:
        name = f"route_transfer_lgbm_{int((1-aux_w)*100)}_{int(aux_w*100)}"
        preds[name] = apply_route(cb, lgbm, masks["transfer_positive"], aux_w)
        add_meta(name, "single_branch", LGBM_NAME, "embryo_transferred_flag == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["transfer_positive"], "medium", "Transfer-positive low-weight LGBM routing")

    for aux_w in [0.2, 0.3]:
        name = f"route_day5_lgbm_{int((1-aux_w)*100)}_{int(aux_w*100)}_diagnostic_only"
        preds[name] = apply_route(cb, lgbm, masks["day5"], aux_w)
        add_meta(name, "single_branch", LGBM_NAME, "transfer_day_5 == 1", f"{1-aux_w:.2f},{aux_w:.2f}", masks["day5"], "high", "Day5 routing diagnostic only")

    combined_specs = {
        "router_conservative_v1": [("donor_egg", "lgbm", 0.4), ("no_transfer", "et", 0.2), ("frozen", "lgbm", 0.2), ("transfer_positive", "lgbm", 0.2)],
        "router_donor_no_transfer_v1": [("donor_egg", "lgbm", 0.5), ("no_transfer", "et", 0.2)],
        "router_donor_frozen_v1": [("donor_egg", "lgbm", 0.5), ("frozen", "lgbm", 0.3)],
        "router_donor_no_transfer_frozen_v1": [("donor_egg", "lgbm", 0.5), ("no_transfer", "et", 0.2), ("frozen", "lgbm", 0.3)],
        "router_donor_no_transfer_frozen_storage_low_v1": [("donor_egg", "lgbm", 0.5), ("no_transfer", "et", 0.2), ("frozen", "lgbm", 0.3), ("storage_only", "et", 0.05)],
        "router_aggressive_diagnostic_only": [("donor_egg", "lgbm", 0.5), ("no_transfer", "et", 0.5), ("storage_only", "et", 0.5), ("frozen", "lgbm", 0.4), ("transfer_positive", "lgbm", 0.3)],
    }
    for name, steps in combined_specs.items():
        affected = pd.Series(False, index=preds.index)
        for mask_name, _, _ in steps:
            affected |= masks[mask_name]
        preds[name] = sequential_router(cb, masks, aux, steps)
        risk = "high" if "aggressive" in name or "storage" in name else "medium"
        condition = " sequential; ".join(f"{m} -> {a}:{w}" for m, a, w in steps)
        add_meta(name, "combined_router", ",".join(sorted({a for _, a, _ in steps})), condition, ";".join(str(s) for s in steps), affected, risk, "Sequential conditional soft routing")

    return preds, pd.DataFrame(metadata)


def scale_warning(pred: pd.Series) -> str:
    if pred.min() < 0 or pred.max() > 1:
        return "out_of_probability_range"
    return ""


def evaluate_results(preds: pd.DataFrame, meta: pd.DataFrame, y: pd.Series, fold_compatible: dict[str, bool]) -> pd.DataFrame:
    cb_auc = compute_auc(y, preds["global_cb_only"])
    global_auc = compute_auc(y, preds[GLOBAL_BLEND_NAME])
    rows = []
    y_values = y.reset_index(drop=True)
    for _, m in meta.iterrows():
        name = m["routing_name"]
        pred = preds[name]
        affected_mask = pred.ne(preds["global_cb_only"])
        affected_positive_rate = y_values[affected_mask].mean() if affected_mask.any() else y_values.mean()
        warning = scale_warning(pred)
        auc = compute_auc(y_values, pred)
        candidate_status = candidate_status_for(name, auc, cb_auc, global_auc, warning, m["risk_level"])
        rows.append(
            {
                **m.to_dict(),
                "oof_auc": auc,
                "delta_vs_catboost": auc - cb_auc,
                "delta_vs_global_cb_lgbm_60_40": auc - global_auc,
                "affected_n": int(affected_mask.sum()) if name != "global_cb_only" else len(preds),
                "affected_rate": float(affected_mask.mean()) if name != "global_cb_only" else 1.0,
                "affected_positive_rate": affected_positive_rate,
                "pred_min": pred.min(),
                "pred_max": pred.max(),
                "pred_mean": pred.mean(),
                "pred_std": pred.std(),
                "scale_warning": warning,
                "candidate_status": candidate_status,
                "note": m["note"] + ("; uses fold-incompatible aux OOF for fold diagnostics" if not route_fold_compatible(name, fold_compatible) else ""),
            }
        )
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False)


def candidate_status_for(name: str, auc: float, cb_auc: float, global_auc: float, warning: str, risk: str) -> str:
    diagnostic = "diagnostic_only" in name or "aggressive" in name
    if auc < cb_auc:
        return "reject"
    if diagnostic or risk == "high":
        return "diagnostic_only"
    if auc >= global_auc + 0.0002 and not warning:
        return "strong_candidate"
    if auc >= global_auc and not warning:
        return "candidate"
    return "hold"


def route_fold_compatible(name: str, fold_compatible: dict[str, bool]) -> bool:
    if "_et_" in name or "no_transfer" in name or "storage" in name:
        return fold_compatible.get(ET_NAME, False)
    return fold_compatible.get(LGBM_NAME, True)


def fold_results(preds: pd.DataFrame, results: pd.DataFrame, y: pd.Series, fold_compatible: dict[str, bool]) -> pd.DataFrame:
    rows = []
    cb = preds["global_cb_only"]
    fold = preds[f"{CB_NAME}__fold"]
    for name in results["routing_name"]:
        pred = preds[name]
        compatible = route_fold_compatible(name, fold_compatible)
        for f in sorted(fold.dropna().unique()):
            mask = fold.eq(f)
            auc = compute_auc(y[mask], pred[mask])
            cb_auc = compute_auc(y[mask], cb[mask])
            rows.append(
                {
                    "routing_name": name,
                    "fold": f,
                    "n": int(mask.sum()),
                    "auc": auc,
                    "delta_vs_cb_fold": auc - cb_auc,
                    "note": "" if compatible else "fold_incompatible_aux_oof",
                }
            )
    return pd.DataFrame(rows)


def subgroup_auc(preds: pd.DataFrame, results: pd.DataFrame, X: pd.DataFrame, raw: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cb_pred = preds["global_cb_only"]
    global_pred = preds[GLOBAL_BLEND_NAME]
    rows = []
    frame = X.copy().reset_index(drop=True)
    frame["ID"] = raw[ID_COLUMN].astype(str).to_numpy()
    frame["y_true"] = y.to_numpy()
    for name in results["routing_name"]:
        frame["oof_pred"] = preds[name].to_numpy()
        for subgroup, value, mask in subgroup_specs(frame):
            sub = frame[mask.fillna(False)]
            if sub.empty:
                continue
            idx = sub.index
            auc = compute_auc(sub["y_true"], sub["oof_pred"]) if sub["y_true"].nunique() > 1 else np.nan
            cb_auc = compute_auc(sub["y_true"], cb_pred.iloc[idx]) if sub["y_true"].nunique() > 1 else np.nan
            global_auc = compute_auc(sub["y_true"], global_pred.iloc[idx]) if sub["y_true"].nunique() > 1 else np.nan
            pos = sub.loc[sub["y_true"].eq(1), "oof_pred"].mean() if sub["y_true"].eq(1).any() else np.nan
            neg = sub.loc[sub["y_true"].eq(0), "oof_pred"].mean() if sub["y_true"].eq(0).any() else np.nan
            rows.append(
                {
                    "routing_name": name,
                    "subgroup": subgroup,
                    "value": value,
                    "n": len(sub),
                    "positive_rate": sub["y_true"].mean(),
                    "auc": auc,
                    "delta_vs_cb": auc - cb_auc if pd.notna(auc) and pd.notna(cb_auc) else np.nan,
                    "delta_vs_global_cb_lgbm_60_40": auc - global_auc if pd.notna(auc) and pd.notna(global_auc) else np.nan,
                    "pred_mean": sub["oof_pred"].mean(),
                    "calibration_gap": sub["oof_pred"].mean() - sub["y_true"].mean(),
                    "separation": pos - neg if pd.notna(pos) and pd.notna(neg) else np.nan,
                    "note": "" if sub["y_true"].nunique() > 1 else "only one class present",
                }
            )
    return pd.DataFrame(rows)


def branch_focus(subgroup: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
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
    safe_names = set(results[results["candidate_status"].isin(["candidate", "strong_candidate", "hold"])]["routing_name"])
    rows = []
    for branch, (subgroup_name, value) in branch_map.items():
        sub = subgroup[subgroup["subgroup"].eq(subgroup_name) & subgroup["value"].astype(str).eq(value)]
        if sub.empty:
            continue
        cb = sub[sub["routing_name"].eq("global_cb_only")]
        glob = sub[sub["routing_name"].eq(GLOBAL_BLEND_NAME)]
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        safe = sub[sub["routing_name"].isin(safe_names)].sort_values("auc", ascending=False, na_position="last")
        cb_auc = cb["auc"].iloc[0] if len(cb) else np.nan
        global_auc = glob["auc"].iloc[0] if len(glob) else np.nan
        best_safe = safe.iloc[0] if len(safe) else best
        rows.append(
            {
                "branch_name": branch,
                "n": int(best["n"]),
                "positive_rate": best["positive_rate"],
                "catboost_auc": cb_auc,
                "global_blend_auc": global_auc,
                "best_routing_name": best["routing_name"],
                "best_routing_auc": best["auc"],
                "delta_best_routing_vs_catboost": best["auc"] - cb_auc if pd.notna(cb_auc) else np.nan,
                "delta_best_routing_vs_global_blend": best["auc"] - global_auc if pd.notna(global_auc) else np.nan,
                "best_safe_routing_name": best_safe["routing_name"],
                "best_safe_routing_auc": best_safe["auc"],
                "delta_best_safe_vs_catboost": best_safe["auc"] - cb_auc if pd.notna(cb_auc) else np.nan,
                "interpretation": "routing improves branch" if pd.notna(cb_auc) and best["auc"] > cb_auc else "no branch gain",
                "next_action": "consider controlled router" if pd.notna(global_auc) and best_safe["auc"] >= global_auc else "monitor",
            }
        )
    return pd.DataFrame(rows)


def scale_diagnostics(preds: pd.DataFrame, results: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    branches = {
        "donor_egg": X["is_donor_egg"].eq(1),
        "no_transfer": X["no_embryo_transfer_flag"].eq(1),
        "storage_only": X["storage_only_flag"].eq(1),
        "frozen": X["is_frozen_embryo"].eq(1),
        "transfer_positive": X["embryo_transferred_flag"].eq(1),
        "day5_transfer": X["transfer_day_5"].eq(1),
    }
    rows = []
    cb = preds["global_cb_only"]
    for name in results["routing_name"]:
        pred = preds[name]
        for branch, mask in branches.items():
            if not pred[mask].ne(cb[mask]).any() and name not in {"global_cb_only", GLOBAL_BLEND_NAME}:
                continue
            mean_shift = pred[mask].mean() - cb[mask].mean()
            std_shift = pred[mask].std() - cb[mask].std()
            warning = ""
            if abs(mean_shift) > 0.03 or pred[mask].min() < 0 or pred[mask].max() > 1 or pred[mask].mean() < 0 or pred[mask].mean() > 1:
                warning = "scale_shift_or_range_warning"
            rows.append(
                {
                    "routing_name": name,
                    "branch_name": branch,
                    "n": int(mask.sum()),
                    "positive_rate": y[mask].mean(),
                    "cb_pred_mean": cb[mask].mean(),
                    "routed_pred_mean": pred[mask].mean(),
                    "mean_shift": mean_shift,
                    "cb_pred_std": cb[mask].std(),
                    "routed_pred_std": pred[mask].std(),
                    "std_shift": std_shift,
                    "pred_min": pred[mask].min(),
                    "pred_max": pred[mask].max(),
                    "scale_warning": warning,
                }
            )
    return pd.DataFrame(rows)


def expert_pool_used(branch_prior: Path) -> pd.DataFrame:
    prior = pd.read_csv(branch_prior) if branch_prior.exists() else pd.DataFrame()
    def prior_row(branch: str, aux: str) -> dict[str, Any]:
        if prior.empty:
            return {}
        sub = prior[prior["branch_name"].eq(branch)]
        if sub.empty:
            return {}
        row = sub.iloc[0].to_dict()
        return row

    specs = [
        ("donor_egg", "is_donor_egg == 1", LGBM_NAME, "LGBM beat CatBoost in donor_egg branch", "0.5 LGBM in donor routers; 0.1-1.0 tested", "medium"),
        ("no_transfer", "no_embryo_transfer_flag == 1", ET_NAME, "ExtraTrees beat CatBoost in no_transfer branch", "0.2 ET in combined routers; 0.1-1.0 tested", "medium"),
        ("storage_only", "storage_only_flag == 1", ET_NAME, "ExtraTrees large storage_only OOF gain but extreme positive-rate risk", "0.02/0.05/0.1 cautious only; high weights diagnostic", "high"),
        ("frozen", "is_frozen_embryo == 1", LGBM_NAME, "Global CB+LGBM blend improved frozen branch", "0.1-0.4 LGBM tested", "medium"),
        ("transfer_positive", "embryo_transferred_flag == 1", LGBM_NAME, "Global CB+LGBM blend improved transfer-positive branch", "0.1-0.4 LGBM tested", "medium"),
        ("day5_transfer", "transfer_day_5 == 1", LGBM_NAME, "Day5 monitored only", "0.2/0.3 diagnostic only", "high"),
    ]
    rows = []
    for branch, condition, aux, reason, weights, risk in specs:
        p = prior_row(branch, aux)
        rows.append(
            {
                "branch_name": branch,
                "condition": condition,
                "base_model": CB_NAME,
                "aux_model": aux,
                "reason_from_model_zoo": reason,
                "catboost_branch_auc": p.get("catboost_auc", np.nan),
                "aux_branch_auc": p.get("best_non_catboost_auc", np.nan),
                "best_prior_blend_auc": p.get("best_blend_auc", np.nan),
                "selected_weights": weights,
                "routing_candidates_tested": "single-branch and combined sequential routers",
                "risk_level": risk,
            }
        )
    return pd.DataFrame(rows)


def candidate_recommendation(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = results.sort_values("oof_auc", ascending=False).reset_index(drop=True)
    for i, row in ordered.iterrows():
        recommend = "no"
        reason = "below global blend or diagnostic/high-risk"
        if row["candidate_status"] == "strong_candidate":
            recommend = "yes"
            reason = "safe routing beats global blend by required margin"
        elif row["candidate_status"] == "candidate":
            recommend = "hold"
            reason = "safe routing reaches global blend but margin is small"
        elif row["candidate_status"] == "hold":
            recommend = "hold"
            reason = "OOF improvement is marginal"
        rows.append(
            {
                "rank": i + 1,
                "routing_name": row["routing_name"],
                "oof_auc": row["oof_auc"],
                "delta_vs_catboost": row["delta_vs_catboost"],
                "delta_vs_global_cb_lgbm_60_40": row["delta_vs_global_cb_lgbm_60_40"],
                "risk_level": row["risk_level"],
                "candidate_status": row["candidate_status"],
                "reason": reason,
                "recommend_submission": recommend,
            }
        )
    return pd.DataFrame(rows)


def optional_sanity(note: str) -> pd.DataFrame:
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


def build_report(tables: dict[str, pd.DataFrame], optional_note: str) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def table(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    results = tables["routing_results"].sort_values("oof_auc", ascending=False)
    best = results.iloc[0]
    safe = results[results["candidate_status"].isin(["candidate", "strong_candidate", "hold"])].sort_values("oof_auc", ascending=False)
    best_safe = safe.iloc[0] if len(safe) else best
    parts = [
        f"<style>{css}</style>",
        "<h1>Branch Soft Routing v1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best routing: {html.escape(str(best['routing_name']))}, OOF AUC {best['oof_auc']:.6f}. Best safe routing: {html.escape(str(best_safe['routing_name']))}, OOF AUC {best_safe['oof_auc']:.6f}.</p>",
        "<h2>Leakage and Prohibited Methods</h2>",
        "<p>Only train OOF predictions and train row-wise branch flags were used. No public-LB-based routing or test-wide post-processing was used.</p>",
        "<h2>Why Branch Soft Routing Was Needed</h2>",
        "<p>Model Zoo showed branch-specific auxiliary signals but no stronger global non-CatBoost model.</p>",
        "<h2>Expert Pool Used</h2>",
        table("routing_expert_pool_used", 20),
        "<h2>Global Reference Blends</h2>",
        table("global_reference", 10),
        "<h2>Single-Branch Routing Results</h2>",
        table("single_branch", 40),
        "<h2>Combined Router Results</h2>",
        table("combined_router", 20),
        "<h2>Subgroup AUC</h2>",
        table("routing_subgroup_auc", 80),
        "<h2>Branch Focus Diagnostics</h2>",
        table("routing_branch_focus_diagnostics", 30),
        "<h2>Scale Diagnostics</h2>",
        table("routing_scale_diagnostics", 80),
        "<h2>Candidate Recommendation</h2>",
        table("routing_candidate_recommendation", 30),
        "<h2>Optional Submission Candidates</h2>",
        f"<p>{html.escape(optional_note)}</p>",
        table("optional_submission_sanity", 10),
        "<h2>Interpretation</h2>",
        "<p>Prefer safe routers that beat the global CB+LGBM blend without storage-only hard routing or scale warnings.</p>",
        "<h2>Next Step Recommendation</h2>",
        "<p>If no safe routing beats the global blend, continue with controlled global ensemble rather than branch routing.</p>",
    ]
    (OUT_DIR / "branch_soft_routing_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTIONAL_SUB_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        train = pd.read_csv(TRAIN_PATH)
        target = get_target_column(train)
        y = train[target].astype(int).reset_index(drop=True)
        X = make_art_features(train).reset_index(drop=True)
        oof, loaded_paths = load_oof()
        oof = oof.set_index("ID").loc[train[ID_COLUMN].astype(str)].reset_index()
        X.index = oof.index

        fold_compatible = {
            LGBM_NAME: set(oof[f"{CB_NAME}__fold"].dropna().unique()) == set(oof[f"{LGBM_NAME}__fold"].dropna().unique()),
            ET_NAME: ET_NAME in oof.columns and set(oof[f"{CB_NAME}__fold"].dropna().unique()) == set(oof[f"{ET_NAME}__fold"].dropna().unique()),
        }
        preds, meta = build_predictions(oof, X)
        results = evaluate_results(preds, meta, y, fold_compatible)
        fold = fold_results(preds, results, y, fold_compatible)
        subgroup = subgroup_auc(preds, results, X, train, y)
        branch = branch_focus(subgroup, results)
        scale = scale_diagnostics(preds, results, X, y)
        expert = expert_pool_used(PROJECT_ROOT / "outputs" / "model_zoo_v1" / "model_zoo_branch_focus_diagnostics.csv")
        recommendation = candidate_recommendation(results)

        best_safe = results[results["candidate_status"].isin(["candidate", "strong_candidate"])].sort_values("oof_auc", ascending=False)
        optional_note = "Optional submissions disabled by default."
        if args.make_optional_submissions:
            if len(best_safe) and (best_safe.iloc[0]["oof_auc"] >= PREVIOUS_GLOBAL_BLEND_AUC + 0.0002 or best_safe.iloc[0]["delta_vs_catboost"] >= 0.0004):
                optional_note = "OOF criteria met, but candidate generation is deferred because this stage intentionally uses OOF-only routing diagnostics."
            else:
                optional_note = "OOF criteria for optional submission were not met."
        else:
            optional_note = "Optional submissions disabled by default; OOF-only diagnostics stage."
        sanity = optional_sanity(optional_note)

        routing_oof = prediction_oof_long(preds, y)
        tables = {
            "routing_results": results,
            "routing_fold_results": fold,
            "routing_subgroup_auc": subgroup,
            "routing_branch_focus_diagnostics": branch,
            "routing_scale_diagnostics": scale,
            "routing_expert_pool_used": expert,
            "routing_candidate_recommendation": recommendation,
            "optional_submission_sanity": sanity,
            "global_reference": results[results["routing_type"].eq("global_reference")],
            "single_branch": results[results["routing_type"].eq("single_branch")],
            "combined_router": results[results["routing_type"].eq("combined_router")],
        }
        save_table(OUT_DIR / "routing_oof_predictions.csv", routing_oof)
        for key in [
            "routing_results",
            "routing_fold_results",
            "routing_subgroup_auc",
            "routing_branch_focus_diagnostics",
            "routing_scale_diagnostics",
            "routing_expert_pool_used",
            "routing_candidate_recommendation",
        ]:
            save_table(OUT_DIR / f"{key}.csv", tables[key])
        save_table(OUT_DIR / "optional_submission_sanity.csv", sanity)
        save_json(
            OUT_DIR / "model_config.json",
            {
                "random_seed": RANDOM_SEED,
                "base_model": CB_NAME,
                "aux_models": [LGBM_NAME, ET_NAME],
                "loaded_oof_files": loaded_paths,
                "fold_compatible": fold_compatible,
                "previous_global_blend_auc": PREVIOUS_GLOBAL_BLEND_AUC,
                "prohibited_methods": PROHIBITED_NOTE,
            },
        )
        build_report(tables, optional_note)

        warnings_list.extend(str(w.message) for w in caught)
        cb_auc = float(results.loc[results["routing_name"].eq("global_cb_only"), "oof_auc"].iloc[0])
        global_auc = float(results.loc[results["routing_name"].eq(GLOBAL_BLEND_NAME), "oof_auc"].iloc[0])
        best = results.iloc[0]
        safe_pool = results[results["candidate_status"].isin(["candidate", "strong_candidate", "hold"])].sort_values("oof_auc", ascending=False)
        best_safe_row = safe_pool.iloc[0] if len(safe_pool) else best
        scale_warnings = scale[scale["scale_warning"].ne("")]
        branch_summary = branch[branch["branch_name"].isin(["donor_egg", "no_transfer", "storage_only", "frozen", "transfer_positive", "day5_transfer"])].to_dict(orient="records")
        log = [
            f"run_time: {datetime.now().isoformat(timespec='seconds')}",
            f"python_version: {platform.python_version()}",
            f"pandas_version: {pd.__version__}",
            f"numpy_version: {np.__version__}",
            f"train_path: {TRAIN_PATH}",
            f"target_column: {target}",
            f"loaded_oof_files: {loaded_paths}",
            f"loaded_model_names: {CB_NAME},{LGBM_NAME},{ET_NAME}",
            f"baseline_catboost_oof_auc: {cb_auc:.6f}",
            f"global_cb_lgbm_60_40_oof_auc: {global_auc:.6f}",
            f"routing_candidates_tested: {len(results)}",
            f"best_routing: {best['routing_name']}",
            f"best_routing_oof_auc: {best['oof_auc']:.6f}",
            f"best_safe_routing: {best_safe_row['routing_name']}",
            f"best_safe_routing_oof_auc: {best_safe_row['oof_auc']:.6f}",
            f"branch_improvements_summary: {branch_summary}",
            f"scale_warnings: {len(scale_warnings)}",
            f"optional_candidate_paths: none",
            f"leakage_prohibited_methods_note: {PROHIBITED_NOTE}",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "branch_soft_routing_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
