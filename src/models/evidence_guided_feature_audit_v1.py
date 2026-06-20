from __future__ import annotations

import argparse
import html
import platform
import re
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
from src.models.controlled_full_cv_v1 import all_cols_from_families, family_lookup
from src.models.model_branch_diagnostics_v1 import subgroup_auc_by_model
from src.models.model_utils import compute_auc, get_categorical_columns, get_target_column, prepare_catboost_frame, save_json, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
CB_OOF_PATH = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "long_oof_predictions.csv"
CB_RESULT_PATH = PROJECT_ROOT / "outputs" / "catboost_long_underfit_check_v1" / "long_cv_results.csv"
LGBM_OOF_PATH = PROJECT_ROOT / "outputs" / "model_branch_diagnostics_v1" / "model_oof_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "evidence_guided_feature_audit_v1"
OPTIONAL_SUB_DIR = OUT_DIR / "optional_submissions"
RANDOM_SEED = 42
OLD_CB_NAME = "CB_long_all_features_depth6"
LGBM_NAME = "LGBM_A_all_features"
OLD_GLOBAL_BLEND_AUC = 0.740462
PROHIBITED_NOTE = (
    "Pseudo-labeling, self-training, target encoding, rank normalization, MICE imputation, "
    "train+test concat, test-wide post-processing, public-LB-based tuning, and any use of test data "
    "for training or feature decisions were not used."
)

AUDIT_CONFIG = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 500,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.2,
    "border_count": 128,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
    "early_stopping_rounds": 120,
}
V2_CONFIGS = {
    "CB_v2_depth7_lr045_l2_6": {
        **AUDIT_CONFIG,
        "iterations": 1600,
        "learning_rate": 0.045,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "bagging_temperature": 0.1,
        "border_count": 167,
        "early_stopping_rounds": 250,
    },
    "CB_v2_depth7_lr035_l2_6": {
        **AUDIT_CONFIG,
        "iterations": 2200,
        "learning_rate": 0.035,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "bagging_temperature": 0.1,
        "border_count": 167,
        "early_stopping_rounds": 300,
    },
    "CB_v2_depth6_lr035_l2_5": {
        **AUDIT_CONFIG,
        "iterations": 1800,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "bagging_temperature": 0.2,
        "border_count": 167,
        "early_stopping_rounds": 250,
    },
}
EVIDENCE_FAMILIES = [
    "time_interval_features",
    "egg_biological_age_proxy",
    "set_blastocyst_day5_features",
    "treatment_time_code_features",
    "treatment_time_code_interactions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-Guided Feature Audit & CatBoost v2")
    parser.add_argument("--quick", action="store_true", help="Use reduced 3-fold/short-iteration smoke run.")
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "long"],
        default="all",
        help="Run all stages, only the selected full audit, or only long v2 from saved audit results.",
    )
    parser.add_argument("--include-optional-v2", action="store_true", help="Run optional depth7 lr035 v2 candidate.")
    parser.add_argument("--make-optional-submissions", action="store_true", help="Create optional submissions only if OOF criteria are met. Not used by default.")
    return parser.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _cat(df: pd.DataFrame, col: str, default: str = "__MISSING__") -> pd.Series:
    if col in df:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def age_to_ord(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        return np.nan
    if "이하" in text:
        return float(nums[0])
    if "이상" in text:
        return float(nums[0])
    return float(sum(nums[:2]) / min(len(nums), 2))


def interval_bin(s: pd.Series) -> pd.Series:
    labels = pd.Series("missing", index=s.index, dtype="object")
    labels.loc[s.lt(0)] = "negative"
    for value in range(0, 6):
        labels.loc[s.eq(value)] = str(value)
    labels.loc[s.ge(6)] = "6plus"
    return labels


def add_evidence_features(base: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = base.copy()
    raw = raw.copy()
    raw.index = out.index
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    def add(name: str, family: str, value: pd.Series, rationale: str, source_type: str, expected: str, risk: str, note: str) -> None:
        out[name] = value
        rows.append(
            {
                "feature_family": family,
                "feature_name": name,
                "clinical_or_literature_rationale": rationale,
                "source_type": source_type,
                "expected_signal": expected,
                "risk_level": risk,
                "implementation_note": note,
            }
        )

    def require(cols: list[str], family: str) -> bool:
        missing = [c for c in cols if c not in raw.columns and c not in out.columns]
        for col in missing:
            skipped.append({"feature_family": family, "source_column": col, "reason": "missing source column"})
        return not missing

    # 6A. Time interval features
    family = "time_interval_features"
    transfer = _num(raw, "배아 이식 경과일")
    mix = _num(raw, "난자 혼합 경과일")
    thaw = _num(raw, "배아 해동 경과일")
    pickup = _num(raw, "난자 채취 경과일")
    interval_specs = {
        "mix_to_transfer_interval": (transfer - mix, ["배아 이식 경과일", "난자 혼합 경과일"]),
        "thaw_to_transfer_interval": (transfer - thaw, ["배아 이식 경과일", "배아 해동 경과일"]),
        "pickup_to_mix_interval": (mix - pickup, ["난자 혼합 경과일", "난자 채취 경과일"]),
        "pickup_to_transfer_interval": (transfer - pickup, ["배아 이식 경과일", "난자 채취 경과일"]),
    }
    for name, (values, cols) in interval_specs.items():
        require(cols, family)
        add(name, family, values, "ART cycle timing may reflect embryo culture, thawing, and transfer process.", "dataset-derived row-wise timing logic", "timing/stage signal", "medium", "Raw row-wise day difference; negative values retained with flags.")
        add(f"{name}_missing", family, values.isna().astype("int8"), "Structural missingness can describe branch-specific ART pathways.", "dataset-derived row-wise timing logic", "missingness signal", "low", "Missing flag only; no imputation statistic.")
        add(f"{name}_negative_flag", family, values.lt(0).astype("int8"), "Negative timing may indicate coding/process branch and should be audited, not dropped.", "dataset-derived row-wise timing logic", "data/process sanity signal", "medium", "Negative values are flagged.")
        if name in {"mix_to_transfer_interval", "thaw_to_transfer_interval"}:
            add(f"{name}_zero_flag", family, values.eq(0).astype("int8"), "Zero interval may represent same-day process branch.", "dataset-derived row-wise timing logic", "same-day process signal", "low", "Boolean row-wise flag.")
        if name in {"mix_to_transfer_interval", "thaw_to_transfer_interval", "pickup_to_transfer_interval"}:
            add(f"{name}_bin", family, interval_bin(values), "Binned timing can capture cleavage/blastocyst stage-like patterns.", "blastocyst / day5 embryo transfer literature", "nonlinear timing signal", "medium", "Categorical bin: missing/negative/0..5/6plus.")

    # 6B. Egg biological age proxy
    family = "egg_biological_age_proxy"
    recipient_age = raw["시술 당시 나이"].map(age_to_ord) if "시술 당시 나이" in raw else out.get("age_ord", pd.Series(np.nan, index=out.index))
    donor_age = raw["난자 기증자 나이"].map(age_to_ord) if "난자 기증자 나이" in raw else pd.Series(np.nan, index=out.index)
    egg_source = _cat(raw, "난자 출처")
    donor_egg = out["is_donor_egg"].eq(1) if "is_donor_egg" in out else egg_source.str.contains("기증", na=False)
    own_egg = out["is_own_egg"].eq(1) if "is_own_egg" in out else egg_source.str.contains("본인", na=False)
    egg_age = pd.Series(np.nan, index=out.index, dtype="float64")
    egg_age.loc[own_egg] = recipient_age.loc[own_egg]
    egg_age.loc[donor_egg] = donor_age.loc[donor_egg]
    source = pd.Series("unknown", index=out.index, dtype="object")
    source.loc[own_egg] = "recipient_age_for_own_egg"
    source.loc[donor_egg & donor_age.notna()] = "donor_age"
    source.loc[donor_egg & donor_age.isna()] = "donor_age_unknown"
    add("recipient_age_ord", family, recipient_age, "Recipient age is a core ART outcome correlate.", "oocyte age / donor oocyte literature", "age gradient", "low", "Ordinal midpoint from age category.")
    add("egg_age_proxy_ord", family, egg_age, "Oocyte biological age may differ from recipient age in donor egg cycles.", "oocyte age / donor oocyte literature", "egg source adjusted age", "medium", "Own egg uses recipient age; donor egg uses donor age if known.")
    add("egg_age_proxy_source", family, source, "Proxy provenance is important because donor age availability varies.", "oocyte age / donor oocyte literature", "source/missingness signal", "low", "Categorical source flag.")
    add("recipient_minus_egg_age_proxy", family, recipient_age - egg_age, "Recipient age and oocyte age can diverge in donor cycles.", "oocyte age / donor oocyte literature", "recipient-oocyte age gap", "medium", "Row-wise difference only.")
    add("egg_age_unknown_flag", family, egg_age.isna().astype("int8"), "Unknown oocyte age can indicate distinct treatment branch.", "dataset-derived row-wise timing logic", "missingness signal", "low", "Boolean missing flag.")
    add("donor_egg_age_known_flag", family, (donor_egg & donor_age.notna()).astype("int8"), "Known donor age may refine donor egg branch ranking.", "oocyte age / donor oocyte literature", "donor egg specificity", "medium", "Boolean flag.")
    add("own_egg_age_proxy_flag", family, own_egg.astype("int8"), "Own egg proxy uses recipient age.", "oocyte age / donor oocyte literature", "egg source branch", "low", "Boolean flag.")
    add("donor_egg_age_proxy_flag", family, donor_egg.astype("int8"), "Donor egg proxy separates recipient age from oocyte age.", "oocyte age / donor oocyte literature", "egg source branch", "low", "Boolean flag.")
    for name, values in {
        "egg_age_proxy_x_donor_egg": egg_age * donor_egg.astype(int),
        "egg_age_proxy_x_own_egg": egg_age * own_egg.astype(int),
        "recipient_age_x_donor_egg": recipient_age * donor_egg.astype(int),
        "recipient_age_x_own_egg": recipient_age * own_egg.astype(int),
        "recipient_minus_egg_age_x_donor_egg": (recipient_age - egg_age) * donor_egg.astype(int),
        "egg_age_proxy_x_transfer_day5": egg_age * out.get("transfer_day_5", pd.Series(0, index=out.index)).fillna(0),
        "egg_age_proxy_x_frozen": egg_age * out.get("is_frozen_embryo", pd.Series(0, index=out.index)).fillna(0),
    }.items():
        add(name, family, values, "Age effect can vary by donor/own egg and transfer branch.", "oocyte age / donor oocyte literature", "branch-specific age gradient", "medium", "Row-wise multiplicative interaction.")

    # 6C. SET / blastocyst / day5
    family = "set_blastocyst_day5_features"
    single_raw = _num(raw, "단일 배아 이식 여부").fillna(0).gt(0)
    transferred_count = _num(raw, "이식된 배아 수")
    single_count = transferred_count.eq(1)
    day5 = out.get("transfer_day_5", pd.Series(0, index=out.index)).fillna(0).eq(1)
    blast = out.get("has_blastocyst", pd.Series(0, index=out.index)).fillna(0).eq(1)
    frozen = out.get("is_frozen_embryo", pd.Series(0, index=out.index)).fillna(0).eq(1)
    fresh = out.get("is_fresh_embryo", pd.Series(0, index=out.index)).fillna(0).eq(1)
    age40 = recipient_age.ge(40)
    age43 = recipient_age.ge(43)
    multi = transferred_count.gt(1)
    specific = _cat(raw, "특정 시술 유형", "")
    add("set_exact_single_transfer_flag", family, (single_raw & single_count).astype("int8"), "Single embryo transfer is a clinically meaningful transfer strategy.", "ASRM embryo transfer guidance", "SET strategy signal", "low", "Requires raw SET flag and transferred count == 1.")
    add("single_transfer_count_flag", family, single_count.astype("int8"), "Transferred embryo count distinguishes SET from multi-transfer.", "ASRM embryo transfer guidance", "transfer count signal", "low", "Boolean count flag.")
    add("set_inconsistent_flag", family, (single_raw ^ single_count).astype("int8"), "SET flag/count inconsistency can mark data branch.", "dataset-derived row-wise timing logic", "data/process signal", "medium", "XOR of raw SET flag and count-derived SET.")
    for name, values in {
        "day5_single_transfer_flag": day5 & single_count,
        "blastocyst_single_transfer_flag": blast & single_count,
        "day5_blastocyst_single_transfer_flag": day5 & blast & single_count,
        "set_x_age_40plus": single_count & age40,
        "set_x_age_43plus": single_count & age43,
        "set_x_donor_egg": single_count & donor_egg,
        "set_x_own_egg": single_count & own_egg,
        "set_x_frozen": single_count & frozen,
        "set_x_fresh": single_count & fresh,
        "multi_embryo_transfer_flag_evidence": multi,
        "multi_embryo_x_day5": multi & day5,
        "multi_embryo_x_age_40plus": multi & age40,
        "colon_in_specific_treatment_flag": specific.str.contains(":", regex=False),
        "slash_in_specific_treatment_flag": specific.str.contains("/", regex=False),
        "explicit_blastocyst_token": blast,
    }.items():
        add(name, family, values.astype("int8"), "Transfer strategy, blastocyst/day5 stage, age, and egg source can interact.", "ASRM embryo transfer guidance", "transfer strategy interaction", "medium", "Boolean row-wise feature.")
    token_count = specific.replace("__MISSING__", "").str.split(r"[/,:+ ]+", regex=True).map(lambda x: len([v for v in x if v]))
    add("specific_treatment_token_count_evidence", family, token_count, "Specific treatment complexity can proxy protocol branch.", "dataset-derived row-wise timing logic", "protocol complexity", "medium", "Target-free token count.")

    # 6D/E. Treatment time code and interactions
    family = "treatment_time_code_features"
    timecode = _cat(raw, "시술 시기 코드")
    add("treatment_time_code_raw_category", family, timecode, "Treatment time code may proxy period/practice pattern without target encoding.", "top-code audit idea, not copied", "period/practice proxy", "medium", "Raw category for CatBoost only.")
    add("treatment_time_code_missing_flag", family, timecode.eq("__MISSING__").astype("int8"), "Missing treatment time code may mark data process branch.", "dataset-derived row-wise timing logic", "missingness signal", "low", "Boolean missing flag.")

    family = "treatment_time_code_interactions"
    interaction_sources = {
        "timecode_x_treatment_type": _cat(raw, "시술 유형"),
        "timecode_x_age_group": _cat(raw, "시술 당시 나이"),
        "timecode_x_egg_source": _cat(raw, "난자 출처"),
        "timecode_x_sperm_source": _cat(raw, "정자 출처"),
        "timecode_x_fresh_frozen_state": pd.Series(np.where(frozen, "frozen", np.where(fresh, "fresh", "unknown")), index=out.index),
        "timecode_x_embryo_usage_type": _cat(raw, "배아 생성 주요 이유"),
        "timecode_x_transfer_day_bin": out.get("transfer_day_raw", pd.Series(np.nan, index=out.index)).map(lambda x: "missing" if pd.isna(x) else ("6plus" if x >= 6 else str(int(x)))),
        "timecode_x_single_embryo": single_count.map({True: "single", False: "not_single"}),
        "timecode_x_donor_egg": donor_egg.map({True: "donor", False: "not_donor"}),
        "timecode_x_frozen": frozen.map({True: "frozen", False: "not_frozen"}),
        "timecode_x_day5": day5.map({True: "day5", False: "not_day5"}),
        "timecode_x_blastocyst_token": blast.map({True: "blast", False: "not_blast"}),
        "timecode_x_pgd_pgs_state": (out.get("has_pgd", pd.Series(0, index=out.index)).fillna(0).astype(str) + "_" + out.get("has_pgs", pd.Series(0, index=out.index)).fillna(0).astype(str)),
        "timecode_x_ovulation_stimulation": _cat(raw, "배란 자극 여부"),
    }
    for name, values in interaction_sources.items():
        add(name, family, timecode + "__" + values.fillna("__MISSING__").astype(str), "Practice period may interact with protocol, branch, and transfer strategy.", "top-code audit idea, not copied", "period-by-protocol proxy", "high", "Target-free categorical interaction; no frequency or target encoding.")

    feature_list = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    return out, feature_list, skipped_df


def load_old_oof(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cb = pd.read_csv(CB_OOF_PATH)
    if "experiment" in cb.columns:
        cb = cb[cb["experiment"].eq("all_features_long_depth6")].copy()
    cb["model_name"] = OLD_CB_NAME
    cb["feature_set"] = "all_features_expanded_transfer"
    cb = cb[["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]]
    row = {
        "model_name": OLD_CB_NAME,
        "feature_set": "all_features_expanded_transfer",
        "iterations": np.nan,
        "learning_rate": np.nan,
        "depth": np.nan,
        "l2_leaf_reg": np.nan,
        "fold_auc_list": "",
        "mean_auc": np.nan,
        "std_auc": np.nan,
        "oof_auc": compute_auc(cb["y_true"], cb["oof_pred"]),
        "best_iteration_list": "",
        "training_time_sec": 0.0,
        "delta_vs_old_catboost": 0.0,
        "delta_vs_old_global_blend": compute_auc(cb["y_true"], cb["oof_pred"]) - OLD_GLOBAL_BLEND_AUC,
        "note": f"Loaded from {CB_OOF_PATH}",
    }
    if CB_RESULT_PATH.exists():
        res = pd.read_csv(CB_RESULT_PATH)
        res = res[res["experiment"].eq("all_features_long_depth6")]
        if not res.empty:
            r = res.iloc[0]
            row.update({"fold_auc_list": r.get("fold_auc_list", ""), "mean_auc": r.get("mean_auc", np.nan), "std_auc": r.get("std_auc", np.nan), "best_iteration_list": r.get("best_iteration_list", "")})
    lgbm_all = pd.read_csv(LGBM_OOF_PATH)
    lgbm = lgbm_all[lgbm_all["model_name"].eq(LGBM_NAME)][["ID", "y_true", "fold", "model_name", "feature_set", "oof_pred"]].copy()
    return cb, lgbm


def run_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cols: list[str],
    model_name: str,
    feature_set: str,
    config: dict[str, Any],
    n_splits: int,
    lookup: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    from catboost import CatBoostClassifier

    cols = [c for c in dict.fromkeys(cols) if c in X.columns]
    cv = StratifiedKFold(n_splits=min(n_splits, int(y.value_counts().min())), shuffle=True, random_state=RANDOM_SEED)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    imp_parts: list[pd.Series] = []
    start = time.time()
    for fold, (tr, va) in enumerate(cv.split(X[cols], y), start=1):
        X_tr, X_va = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy()
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        X_tr, cat_cols = prepare_catboost_frame(X_tr)
        X_va, _ = prepare_catboost_frame(X_va)
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model = CatBoostClassifier(**config)
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict_proba(X_va)[:, 1]
        fold_aucs.append(compute_auc(y_va, pred))
        best_iterations.append(int(model.get_best_iteration() or config["iterations"]))
        oof_parts.append(pd.DataFrame({"ID": X.index[va].astype(str), "y_true": y_va.to_numpy(), "fold": fold, "model_name": model_name, "feature_set": feature_set, "oof_pred": pred}))
        imp_parts.append(pd.Series(model.get_feature_importance(), index=cols, name=f"fold_{fold}"))
    oof = pd.concat(oof_parts, ignore_index=True)
    imp = pd.concat(imp_parts, axis=1)
    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": imp.index,
            "importance_mean": imp.mean(axis=1).to_numpy(),
            "importance_std": imp.std(axis=1).fillna(0).to_numpy(),
            "rank": np.arange(1, len(imp) + 1),
            "feature_family": [lookup.get(c, "unknown") for c in imp.index],
            "evidence_family": [lookup.get(c, "") if str(lookup.get(c, "")).startswith("evidence_") else "" for c in imp.index],
            "note": "CatBoost feature importance",
        }
    ).sort_values("importance_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    row = {
        "model_name": model_name,
        "feature_set": feature_set,
        "iterations": config["iterations"],
        "learning_rate": config["learning_rate"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "fold_auc_list": ",".join(f"{v:.6f}" for v in fold_aucs),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": compute_auc(oof["y_true"], oof["oof_pred"]),
        "best_iteration_list": ",".join(str(v) for v in best_iterations),
        "training_time_sec": round(time.time() - start, 3),
        "delta_vs_old_catboost": np.nan,
        "delta_vs_old_global_blend": np.nan,
        "note": "CatBoost CV; train-only row-wise evidence features",
    }
    return row, oof, importance


def family_lookup_with_evidence(base_families: dict[str, list[str]], feature_list: pd.DataFrame) -> dict[str, str]:
    families = {k: list(v) for k, v in base_families.items()}
    for fam, sub in feature_list.groupby("feature_family"):
        families[f"evidence_{fam}"] = sub["feature_name"].tolist()
    return family_lookup(families)


def build_feature_sets(base_cols: list[str], feature_list: pd.DataFrame) -> dict[str, list[str]]:
    fam_cols = {fam: sub["feature_name"].tolist() for fam, sub in feature_list.groupby("feature_family")}
    sets = {"baseline": base_cols}
    for fam in EVIDENCE_FAMILIES:
        sets[f"plus_{fam}"] = base_cols + fam_cols.get(fam, [])
    all_evidence = base_cols + [c for fam in EVIDENCE_FAMILIES for c in fam_cols.get(fam, [])]
    sets["all_evidence_features"] = all_evidence
    for fam in EVIDENCE_FAMILIES:
        sets[f"all_evidence_minus_{fam}"] = base_cols + [c for f in EVIDENCE_FAMILIES if f != fam for c in fam_cols.get(f, [])]
    return {k: list(dict.fromkeys(v)) for k, v in sets.items()}


def run_feature_search(X: pd.DataFrame, y: pd.Series, feature_sets: dict[str, list[str]], config: dict[str, Any], n_splits: int, lookup: dict[str, str], quick: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    aucs: dict[str, float] = {}
    search_order = [
        "baseline",
        "plus_treatment_time_code_interactions",
        "all_evidence_features",
        "plus_set_blastocyst_day5_features",
        "plus_egg_biological_age_proxy",
    ]
    if not quick:
        search_order = list(dict.fromkeys(search_order))
    for fs in search_order:
        print(f"[evidence_audit] audit start: {fs}", flush=True)
        row, _, _ = run_catboost_cv(X, y, feature_sets[fs], f"CB_audit_depth6_500__{fs}", fs, config, n_splits, lookup)
        print(f"[evidence_audit] audit done: {fs} oof_auc={row['oof_auc']:.6f}", flush=True)
        aucs[fs] = row["oof_auc"]
        rows.append(
            {
                "step": "family_ablation",
                "candidate_family": fs.replace("plus_", ""),
                "included_families": fs,
                "excluded_families": "",
                "model_name": row["model_name"],
                "oof_auc": row["oof_auc"],
                "delta_vs_previous_step": np.nan,
                "delta_vs_baseline": row["oof_auc"] - aucs.get("baseline", row["oof_auc"]),
                "mean_fold_auc": row["mean_auc"],
                "std_fold_auc": row["std_auc"],
                "feature_count": len(feature_sets[fs]),
                "categorical_count": len(get_categorical_columns(X[feature_sets[fs]])),
                "decision": decision_from_delta(row["oof_auc"] - aucs.get("baseline", row["oof_auc"])),
                "note": "Selected family-level audit set requested by user.",
            }
        )
    if not quick:
        return pd.DataFrame(rows), aucs
    selected: list[str] = []
    current_auc = aucs["baseline"]
    remaining = EVIDENCE_FAMILIES.copy()
    step = 0
    while remaining:
        step += 1
        candidates = []
        for fam in remaining:
            fs_cols = list(dict.fromkeys(feature_sets["baseline"] + [c for f in selected + [fam] for c in feature_sets[f"plus_{f}"] if c not in feature_sets["baseline"]]))
            temp_name = f"greedy_step{step}_{fam}"
            row, _, _ = run_catboost_cv(X, y, fs_cols, f"CB_audit_depth6_500__{temp_name}", temp_name, config, n_splits, lookup)
            candidates.append((fam, row, len(fs_cols)))
            if quick and step >= 1:
                # In quick mode, one greedy step is enough to verify automation without excessive runtime.
                pass
        best_fam, best_row, feature_count = max(candidates, key=lambda x: x[1]["oof_auc"])
        delta = best_row["oof_auc"] - current_auc
        decision = "select" if delta >= 0.0001 else "stop"
        rows.append(
            {
                "step": f"greedy_{step}",
                "candidate_family": best_fam,
                "included_families": ",".join(selected + [best_fam]),
                "excluded_families": ",".join([f for f in remaining if f != best_fam]),
                "model_name": best_row["model_name"],
                "oof_auc": best_row["oof_auc"],
                "delta_vs_previous_step": delta,
                "delta_vs_baseline": best_row["oof_auc"] - aucs["baseline"],
                "mean_fold_auc": best_row["mean_auc"],
                "std_fold_auc": best_row["std_auc"],
                "feature_count": feature_count,
                "categorical_count": np.nan,
                "decision": decision,
                "note": "Greedy selection uses train OOF only.",
            }
        )
        if decision != "select":
            break
        selected.append(best_fam)
        remaining.remove(best_fam)
        current_auc = best_row["oof_auc"]
        if quick:
            break
    return pd.DataFrame(rows), aucs


def decision_from_delta(delta: float) -> str:
    if delta >= 0.00015:
        return "useful"
    if delta >= 0.00005:
        return "weak_monitor"
    return "reject"


def make_blends(oof_all: pd.DataFrame, cv_results: pd.DataFrame) -> pd.DataFrame:
    mat = oof_all.pivot_table(index="ID", columns="model_name", values="oof_pred")
    y = oof_all.drop_duplicates("ID").set_index("ID")["y_true"].loc[mat.index]
    cb_v2_rows = cv_results[cv_results["model_name"].str.startswith("CB_v2", na=False)].sort_values("oof_auc", ascending=False)
    if cb_v2_rows.empty:
        return pd.DataFrame()
    best_v2 = cb_v2_rows.iloc[0]["model_name"]
    rows = []
    for aux_name, weights in [
        (LGBM_NAME, [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]),
        (OLD_CB_NAME, [(0.5, 0.5)]),
    ]:
        if aux_name not in mat.columns:
            continue
        for w1, w2 in weights:
            names = [best_v2, aux_name]
            pair = mat[names].dropna()
            pred = w1 * pair[best_v2] + w2 * pair[aux_name]
            auc = compute_auc(y.loc[pair.index], pred)
            rows.append({"blend_name": f"{best_v2}__{aux_name}__{w1}_{w2}", "base_models": ",".join(names), "weights": f"{w1},{w2}", "oof_auc": auc, "delta_vs_old_catboost": auc - cv_results.loc[cv_results["model_name"].eq(OLD_CB_NAME), "oof_auc"].iloc[0], "delta_vs_old_global_blend_0_740462": auc - OLD_GLOBAL_BLEND_AUC, "note": "OOF-only diagnostic blend"})
    if all(name in mat.columns for name in [best_v2, OLD_CB_NAME, LGBM_NAME]):
        for weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.6, 0.2, 0.2)]:
            names = [best_v2, OLD_CB_NAME, LGBM_NAME]
            pair = mat[names].dropna()
            pred = sum(w * pair[n] for w, n in zip(weights, names))
            auc = compute_auc(y.loc[pair.index], pred)
            rows.append({"blend_name": f"{best_v2}__oldCB__LGBM__{'_'.join(str(w) for w in weights)}", "base_models": ",".join(names), "weights": ",".join(str(w) for w in weights), "oof_auc": auc, "delta_vs_old_catboost": auc - cv_results.loc[cv_results["model_name"].eq(OLD_CB_NAME), "oof_auc"].iloc[0], "delta_vs_old_global_blend_0_740462": auc - OLD_GLOBAL_BLEND_AUC, "note": "OOF-only diagnostic blend"})
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False)


def correlation_table(oof_all: pd.DataFrame) -> pd.DataFrame:
    mat = oof_all.pivot_table(index="ID", columns="model_name", values="oof_pred")
    rows = []
    for a in mat.columns:
        for b in mat.columns:
            if a >= b:
                continue
            pair = mat[[a, b]].dropna()
            pearson = pair[a].corr(pair[b]) if len(pair) else np.nan
            rows.append({"model_a": a, "model_b": b, "n": len(pair), "pearson": pearson, "spearman": spearmanr(pair[a], pair[b]).correlation if len(pair) else np.nan, "interpretation": corr_interpretation(pearson)})
    return pd.DataFrame(rows)


def corr_interpretation(c: float) -> str:
    if pd.isna(c):
        return "unknown"
    if c >= 0.995:
        return "near identical"
    if c >= 0.990:
        return "weak diversity"
    if c >= 0.970:
        return "possible ensemble value"
    return "large diversity; verify performance"


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
        old = sub[sub["model_name"].eq(OLD_CB_NAME)]
        v2 = sub[sub["model_name"].str.startswith("CB_v2", na=False)]
        blends = sub[sub["model_name"].str.startswith("blend", na=False)]
        best = sub.sort_values("auc", ascending=False, na_position="last").iloc[0]
        old_auc = old["auc"].iloc[0] if len(old) else np.nan
        best_v2_auc = v2["auc"].max() if len(v2) else np.nan
        blend_auc = blends["auc"].max() if len(blends) else np.nan
        rows.append({"branch_name": branch, "n": int(best["n"]), "positive_rate": best["positive_rate"], "old_catboost_auc": old_auc, "best_v2_auc": best_v2_auc, "best_blend_auc": blend_auc, "best_model_by_auc": best["model_name"], "delta_best_v2_vs_old": best_v2_auc - old_auc if pd.notna(best_v2_auc) and pd.notna(old_auc) else np.nan, "delta_best_blend_vs_old": blend_auc - old_auc if pd.notna(blend_auc) and pd.notna(old_auc) else np.nan, "interpretation": "v2/blend improves branch" if pd.notna(best_v2_auc) and pd.notna(old_auc) and best_v2_auc > old_auc else "monitor"})
    return pd.DataFrame(rows)


def candidate_recommendation(cv_results: pd.DataFrame, blends: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = cv_results[cv_results["model_name"].str.startswith("CB_v2", na=False)].copy()
    candidates = candidates.rename(columns={"model_name": "model_or_blend_name"})
    candidates["risk_level"] = "medium"
    candidates["candidate_status"] = candidates["oof_auc"].map(lambda x: status(x))
    candidates["recommend_submission"] = candidates["candidate_status"].map(lambda s: "yes" if s == "strong_candidate" else ("hold" if s in {"candidate", "hold"} else "no"))
    candidates["reason"] = "CatBoost v2 OOF criterion"
    for _, r in candidates.iterrows():
        rows.append({"rank": 0, "model_or_blend_name": r["model_or_blend_name"], "oof_auc": r["oof_auc"], "delta_vs_old_catboost": r["delta_vs_old_catboost"], "delta_vs_old_global_blend_0_740462": r["delta_vs_old_global_blend"], "feature_set": r["feature_set"], "risk_level": r["risk_level"], "candidate_status": r["candidate_status"], "recommend_submission": r["recommend_submission"], "reason": r["reason"]})
    for _, r in blends.iterrows():
        rows.append({"rank": 0, "model_or_blend_name": r["blend_name"], "oof_auc": r["oof_auc"], "delta_vs_old_catboost": r["delta_vs_old_catboost"], "delta_vs_old_global_blend_0_740462": r["delta_vs_old_global_blend_0_740462"], "feature_set": r["base_models"], "risk_level": "medium", "candidate_status": status(r["oof_auc"]), "recommend_submission": "yes" if status(r["oof_auc"]) == "strong_candidate" else ("hold" if status(r["oof_auc"]) in {"candidate", "hold"} else "no"), "reason": "OOF-only diagnostic blend criterion"})
    out = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def status(auc: float) -> str:
    if auc >= OLD_GLOBAL_BLEND_AUC + 0.00025:
        return "strong_candidate"
    if auc >= OLD_GLOBAL_BLEND_AUC or auc >= 0.740185 + 0.00035:
        return "candidate"
    if auc >= 0.740185:
        return "hold"
    return "reject"


def optional_sanity(note: str) -> pd.DataFrame:
    return pd.DataFrame([{"candidate": "not_created", "path": "", "row_count_equals_sample": np.nan, "id_order_matches_sample": np.nan, "prediction_has_no_nan": np.nan, "prediction_min": np.nan, "prediction_max": np.nan, "prediction_mean": np.nan, "prediction_std": np.nan, "duplicated_id_count": np.nan, "note": note}])


def build_report(tables: dict[str, pd.DataFrame], optional_note: str) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"

    def t(name: str, n: int = 30) -> str:
        return tables.get(name, pd.DataFrame()).head(n).to_html(index=False, escape=True, border=0)

    cv = tables["catboost_v2_cv_results"].sort_values("oof_auc", ascending=False)
    best = cv.iloc[0]
    parts = [
        f"<style>{css}</style>",
        "<h1>Evidence-Guided Feature Audit & CatBoost v2</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Best single model in this run: {html.escape(str(best['model_name']))}, OOF AUC {best['oof_auc']:.6f}.</p>",
        "<h2>Leakage and Prohibited Methods</h2>",
        "<p>No pseudo-labeling, target encoding, rank normalization, MICE imputation, train+test concat, test-wide post-processing, or public-LB-based tuning was used.</p>",
        "<h2>Why Evidence-Guided Feature Audit Was Needed</h2>",
        "<p>Branch-proxy, model zoo, and soft routing did not produce a safer improvement over the old global blend, so this stage audits small row-wise feature families with clinical/process rationale.</p>",
        "<h2>Literature / Clinical Rationale Table</h2>", t("evidence_rationale_table", 60),
        "<h2>Feature Families Created</h2>", t("evidence_feature_list", 80),
        "<h2>Feature Search Results</h2>", t("feature_search_results", 60),
        "<h2>Family-Level Ablation</h2>", t("feature_family_ablation_results", 60),
        "<h2>CatBoost v2 CV Results</h2>", t("catboost_v2_cv_results", 20),
        "<h2>Diagnostic Blend Results</h2>", t("catboost_v2_diagnostic_blend_results", 30),
        "<h2>Subgroup AUC</h2>", t("catboost_v2_subgroup_auc", 80),
        "<h2>Branch Focus Diagnostics</h2>", t("catboost_v2_branch_focus_diagnostics", 30),
        "<h2>Feature Importance</h2>", t("catboost_v2_feature_importance", 80),
        "<h2>OOF Correlation</h2>", t("catboost_v2_oof_correlation", 30),
        "<h2>Candidate Recommendation</h2>", t("candidate_recommendation", 30),
        "<h2>Optional Submission Candidates</h2>",
        f"<p>{html.escape(optional_note)}</p>",
        "<h2>Interpretation</h2>",
        "<p>Promote only feature families with meaningful OOF gain and stable branch behavior into a full CatBoost v2 run.</p>",
        "<h2>Next Step Recommendation</h2>",
        "<p>If quick mode was used, run the same script without --quick before making any submission candidate.</p>",
    ]
    (OUT_DIR / "evidence_feature_audit_report.html").write_text("\n".join(parts), encoding="utf-8")


def select_feature_set_from_audit(search_results: pd.DataFrame) -> tuple[str, list[str]]:
    family_rows = search_results[search_results["step"].eq("family_ablation")].copy()
    if family_rows.empty:
        return "baseline", []
    family_rows["delta_vs_baseline"] = pd.to_numeric(family_rows["delta_vs_baseline"], errors="coerce")
    family_rows["oof_auc"] = pd.to_numeric(family_rows["oof_auc"], errors="coerce")
    alive = family_rows[
        family_rows["included_families"].ne("baseline")
        & family_rows["delta_vs_baseline"].ge(0.00015)
    ].sort_values("oof_auc", ascending=False)
    if alive.empty:
        return "baseline", []
    selected_key = str(alive.iloc[0]["included_families"])
    if selected_key.startswith("plus_"):
        return selected_key, [selected_key.replace("plus_", "")]
    if selected_key == "all_evidence_features":
        return selected_key, EVIDENCE_FAMILIES.copy()
    return selected_key, [str(alive.iloc[0]["candidate_family"])]


def build_audit_only_report(feature_list: pd.DataFrame, search_results: pd.DataFrame, optional_note: str) -> None:
    css = "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}h1,h2{color:#1f3b4d}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:6px 8px}th{background:#f2f5f7}.note{background:#fff8dc;border-left:4px solid #c99700;padding:10px}"
    selected_key, selected_families = select_feature_set_from_audit(search_results)
    parts = [
        f"<style>{css}</style>",
        "<h1>Evidence-Guided Feature Audit v1 - Stage 1</h1>",
        f"<p class='note'>{html.escape(PROHIBITED_NOTE)}</p>",
        "<h2>Executive Summary</h2>",
        f"<p>Selected feature set for long v2: {html.escape(selected_key)}. Selected families: {html.escape(','.join(selected_families) if selected_families else 'none')}.</p>",
        "<h2>Feature Family Audit</h2>",
        search_results.to_html(index=False, escape=True, border=0),
        "<h2>Evidence Feature List</h2>",
        feature_list.head(120).to_html(index=False, escape=True, border=0),
        "<h2>Optional Submission Candidates</h2>",
        f"<p>{html.escape(optional_note)}</p>",
    ]
    (OUT_DIR / "evidence_feature_audit_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTIONAL_SUB_DIR.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    n_splits = 3 if args.quick else 5
    audit_config = dict(AUDIT_CONFIG)
    v2_configs = {k: dict(v) for k, v in V2_CONFIGS.items()}
    if args.quick:
        audit_config.update({"iterations": 90, "early_stopping_rounds": 30})
        for cfg in v2_configs.values():
            cfg.update({"iterations": 140, "early_stopping_rounds": 40})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PerformanceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        train = pd.read_csv(TRAIN_PATH)
        target = get_target_column(train)
        y = train[target].astype(int)
        X_base = make_art_features(train)
        X_base.index = train[ID_COLUMN].astype(str)
        X, feature_list, skipped_features = add_evidence_features(X_base, train)
        base_families = save_feature_families(OUT_DIR / "feature_families_base.json", X_base.columns.tolist())
        base_cols = all_cols_from_families(base_families, X_base)
        feature_sets = build_feature_sets(base_cols, feature_list)
        lookup = family_lookup_with_evidence(base_families, feature_list)
        save_table(OUT_DIR / "evidence_feature_list.csv", feature_list)
        save_table(OUT_DIR / "evidence_rationale_table.csv", feature_list[["feature_family", "feature_name", "clinical_or_literature_rationale", "source_type", "expected_signal", "risk_level", "implementation_note"]])

        if args.stage == "long":
            search_path = OUT_DIR / "feature_search_results.csv"
            if not search_path.exists():
                raise FileNotFoundError(f"Missing full audit results. Run first: python -m src.models.evidence_guided_feature_audit_v1 --stage audit")
            search_results = pd.read_csv(search_path)
            family_ablation = search_results[search_results["step"].eq("family_ablation")].copy()
        else:
            search_results, search_aucs = run_feature_search(X, y, feature_sets, audit_config, n_splits, lookup, args.quick)
            family_ablation = search_results[search_results["step"].eq("family_ablation")].copy()
            save_table(OUT_DIR / "feature_search_results.csv", search_results)
            save_table(OUT_DIR / "feature_family_ablation_results.csv", family_ablation)
            if args.stage == "audit":
                selected_key, selected_families = select_feature_set_from_audit(search_results)
                save_json(
                    OUT_DIR / "model_config.json",
                    {
                        "stage": "audit",
                        "cv": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
                        "audit_config": audit_config,
                        "selected_feature_set": selected_key,
                        "selected_families": selected_families,
                        "judgement": "+0.00015 useful, +0.00005~+0.00015 hold, <=0 reject",
                        "prohibited_methods": PROHIBITED_NOTE,
                    },
                )
                optional_note = "Audit-only stage; no optional submission candidate can be created."
                build_audit_only_report(feature_list, search_results, optional_note)
                warnings_list.extend(str(w.message) for w in caught)
                log = [
                    f"run_time: {datetime.now().isoformat(timespec='seconds')}",
                    f"python_version: {platform.python_version()}",
                    f"pandas_version: {pd.__version__}",
                    f"numpy_version: {np.__version__}",
                    f"train_path: {TRAIN_PATH}",
                    f"target_column: {target}",
                    f"cv_setting: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
                    f"stage: audit",
                    f"feature_search_results_summary: {search_results.sort_values('oof_auc', ascending=False).to_dict(orient='records')}",
                    f"selected_feature_set: {selected_key}",
                    f"selected_feature_families: {selected_families}",
                    f"prohibited_methods_note: {PROHIBITED_NOTE}",
                    "warnings/errors:",
                ]
                log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
                (OUT_DIR / "evidence_feature_audit_log.txt").write_text("\n".join(log), encoding="utf-8")
                return

        selected_feature_set_key, selected_families = select_feature_set_from_audit(search_results)
        selected_cols = feature_sets.get(selected_feature_set_key, feature_sets["baseline"])
        selected_feature_set_name = selected_feature_set_key

        old_cb, lgbm = load_old_oof(train)
        result_rows = []
        oof_frames = [old_cb, lgbm]
        imp_frames = []
        old_auc = compute_auc(old_cb["y_true"], old_cb["oof_pred"])
        result_rows.append({"model_name": OLD_CB_NAME, "feature_set": "all_features_expanded_transfer", "iterations": np.nan, "learning_rate": np.nan, "depth": np.nan, "l2_leaf_reg": np.nan, "fold_auc_list": "", "mean_auc": np.nan, "std_auc": np.nan, "oof_auc": old_auc, "best_iteration_list": "", "training_time_sec": 0.0, "delta_vs_old_catboost": 0.0, "delta_vs_old_global_blend": old_auc - OLD_GLOBAL_BLEND_AUC, "note": "Loaded baseline OOF"})
        completed = [OLD_CB_NAME, LGBM_NAME, "feature_search"]
        skipped = []
        v2_names = ["CB_v2_depth7_lr045_l2_6", "CB_v2_depth6_lr035_l2_5"]
        if args.include_optional_v2 and not args.quick:
            v2_names.append("CB_v2_depth7_lr035_l2_6")
        else:
            skipped.append("CB_v2_depth7_lr035_l2_6: optional not requested or quick mode")
        for name in v2_names:
            row, oof, imp = run_catboost_cv(X, y, selected_cols, name, selected_feature_set_name, v2_configs[name], n_splits, lookup)
            row["delta_vs_old_catboost"] = row["oof_auc"] - old_auc
            row["delta_vs_old_global_blend"] = row["oof_auc"] - OLD_GLOBAL_BLEND_AUC
            result_rows.append(row)
            oof_frames.append(oof)
            imp_frames.append(imp)
            completed.append(name)

        cv_results = pd.DataFrame(result_rows).sort_values("oof_auc", ascending=False)
        oof_all = pd.concat(oof_frames, ignore_index=True)
        blends = make_blends(oof_all, cv_results)
        blend_oof_frames = []
        mat = oof_all.pivot_table(index="ID", columns="model_name", values="oof_pred")
        y_map = oof_all.drop_duplicates("ID").set_index("ID")["y_true"]
        for _, row in blends.iterrows():
            names = row["base_models"].split(",")
            weights = [float(w) for w in row["weights"].split(",")]
            pair = mat[names].dropna()
            pred = sum(w * pair[n] for w, n in zip(weights, names))
            blend_oof_frames.append(pd.DataFrame({"ID": pair.index, "y_true": y_map.loc[pair.index].to_numpy(), "fold": 0, "model_name": "blend_" + row["blend_name"], "feature_set": "diagnostic_blend", "oof_pred": pred.to_numpy()}))
        oof_for_subgroup = pd.concat([oof_all] + blend_oof_frames, ignore_index=True) if blend_oof_frames else oof_all
        subgroup = subgroup_auc_by_model(X, train, oof_for_subgroup, pd.DataFrame())
        branch = branch_focus(subgroup)
        importance = pd.concat(imp_frames, ignore_index=True) if imp_frames else pd.DataFrame()
        corr = correlation_table(oof_all)
        recommendation = candidate_recommendation(cv_results, blends)
        optional_note = "Optional submissions disabled by default; OOF criteria must be confirmed on full 5-fold run."
        sanity = optional_sanity(optional_note)

        tables = {
            "evidence_feature_list": feature_list,
            "evidence_rationale_table": feature_list,
            "feature_search_results": search_results,
            "feature_family_ablation_results": family_ablation,
            "catboost_v2_cv_results": cv_results,
            "catboost_v2_oof_predictions": oof_all,
            "catboost_v2_subgroup_auc": subgroup,
            "catboost_v2_branch_focus_diagnostics": branch,
            "catboost_v2_feature_importance": importance,
            "catboost_v2_oof_correlation": corr,
            "catboost_v2_diagnostic_blend_results": blends,
            "candidate_recommendation": recommendation,
        }
        for key, df in tables.items():
            save_table(OUT_DIR / f"{key}.csv", df)
        save_table(OUT_DIR / "optional_submission_sanity.csv", sanity)
        save_json(OUT_DIR / "model_config.json", {"cv": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})", "audit_config": audit_config, "v2_configs": {k: v2_configs[k] for k in v2_names}, "selected_families": selected_families, "selected_feature_set": selected_feature_set_name, "quick_mode": args.quick, "prohibited_methods": PROHIBITED_NOTE})
        build_report(tables, optional_note)

        warnings_list.extend(str(w.message) for w in caught)
        import catboost
        import sklearn

        best_single = cv_results[cv_results["model_name"].str.startswith("CB_v2", na=False)].sort_values("oof_auc", ascending=False).head(1)
        best_blend = blends.sort_values("oof_auc", ascending=False).head(1) if not blends.empty else pd.DataFrame()
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
            f"cv_setting: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_SEED})",
            f"baseline_catboost_oof_auc: {old_auc:.6f}",
            f"previous_global_blend_oof_auc: {OLD_GLOBAL_BLEND_AUC:.6f}",
            f"prohibited_methods_note: {PROHIBITED_NOTE}",
            f"feature_families_created: {EVIDENCE_FAMILIES}",
            f"skipped_features_and_reasons: {skipped_features.to_dict(orient='records') if not skipped_features.empty else []}",
            f"feature_search_results_summary: {search_results.sort_values('oof_auc', ascending=False).head(5).to_dict(orient='records')}",
            f"best_feature_family: {family_ablation.sort_values('oof_auc', ascending=False).iloc[0]['candidate_family']}",
            f"selected_feature_families: {selected_families}",
            f"catboost_v2_model_settings: {{k: v2_configs[k] for k in {v2_names}}}",
            f"completed_experiments: {','.join(completed)}",
            f"skipped_experiments: {','.join(skipped) if skipped else 'none'}",
            f"best_single_model: {best_single.iloc[0]['model_name'] if len(best_single) else 'none'}",
            f"best_single_oof_auc: {best_single.iloc[0]['oof_auc'] if len(best_single) else np.nan}",
            f"best_diagnostic_blend: {best_blend.iloc[0]['blend_name'] if len(best_blend) else 'none'}",
            f"best_diagnostic_blend_oof_auc: {best_blend.iloc[0]['oof_auc'] if len(best_blend) else np.nan}",
            f"subgroup_improvement_summary: {branch.to_dict(orient='records')}",
            f"feature_importance_summary: {importance.head(20).to_dict(orient='records') if not importance.empty else []}",
            "optional_candidate_paths: none",
            "warnings/errors:",
        ]
        log.extend(f"- {w}" for w in warnings_list) if warnings_list else log.append("- none")
        (OUT_DIR / "evidence_feature_audit_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
