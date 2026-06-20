from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "임신 성공 여부"
ID_COLUMN = "ID"
MISSING_CATEGORY = "__MISSING__"


AGE_COL = "시술 당시 나이"
TREATMENT_COL = "시술 유형"
SPECIFIC_TREATMENT_COL = "특정 시술 유형"
EGG_SOURCE_COL = "난자 출처"
SPERM_SOURCE_COL = "정자 출처"
EGG_DONOR_AGE_COL = "난자 기증자 나이"
SPERM_DONOR_AGE_COL = "정자 기증자 나이"
DONOR_EMBRYO_COL = "기증 배아 사용 여부"
SURROGACY_COL = "대리모 여부"
FROZEN_EMBRYO_COL = "동결 배아 사용 여부"
FRESH_EMBRYO_COL = "신선 배아 사용 여부"
EMBRYO_THAW_COUNT_COL = "해동된 배아 수"
OOCYTE_THAW_COUNT_COL = "해동 난자 수"
EMBRYO_THAW_DAY_COL = "배아 해동 경과일"
OOCYTE_THAW_DAY_COL = "난자 해동 경과일"
TRANSFER_COUNT_COL = "이식된 배아 수"
ICSI_TRANSFER_COUNT_COL = "미세주입 배아 이식 수"
SINGLE_TRANSFER_COL = "단일 배아 이식 여부"
TRANSFER_DAY_COL = "배아 이식 경과일"
EMBRYO_REASON_COL = "배아 생성 주요 이유"


FUNNEL_COUNT_COLUMNS = [
    "수집된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "총 생성 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
]

HISTORY_COLUMNS = {
    "총 시술 횟수": "total_treatment_count_ord",
    "클리닉 내 총 시술 횟수": "clinic_treatment_count_ord",
    "IVF 시술 횟수": "ivf_treatment_count_ord",
    "DI 시술 횟수": "di_treatment_count_ord",
    "총 임신 횟수": "total_pregnancy_count_ord",
    "IVF 임신 횟수": "ivf_pregnancy_count_ord",
    "DI 임신 횟수": "di_pregnancy_count_ord",
    "총 출산 횟수": "total_birth_count_ord",
    "IVF 출산 횟수": "ivf_birth_count_ord",
    "DI 출산 횟수": "di_birth_count_ord",
}

CAUSE_COLUMNS = [
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna(MISSING_CATEGORY).astype(str).str.strip().replace("", MISSING_CATEGORY)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(MISSING_CATEGORY, index=df.index, dtype="object")
    return normalize_text(df[col])


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.gt(0)))


def bin_count(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series("missing", index=series.index, dtype="object")
    out[x.eq(0)] = "0"
    out[x.eq(1)] = "1"
    out[x.between(2, 3, inclusive="both")] = "2-3"
    out[x.between(4, 5, inclusive="both")] = "4-5"
    out[x.between(6, 10, inclusive="both")] = "6-10"
    out[x.between(11, 15, inclusive="both")] = "11-15"
    out[x.between(16, 20, inclusive="both")] = "16-20"
    out[x.ge(21)] = "21+"
    return out


def map_age_ordinal(series: pd.Series) -> pd.Series:
    text = normalize_text(series)
    mapping = {
        "만18-34세": 0,
        "만35-37세": 1,
        "만38-39세": 2,
        "만40-42세": 3,
        "만43-44세": 4,
        "만45-50세": 5,
    }
    return text.map(mapping).fillna(-1).astype("int16")


def map_count_category(series: pd.Series) -> pd.Series:
    text = normalize_text(series)
    out = pd.Series(-1, index=series.index, dtype="int16")
    for value in range(0, 6):
        out[text.str.contains(fr"{value}\s*회", regex=True, na=False)] = value
    out[text.str.contains(r"6\s*회\s*이상|6회 이상", regex=True, na=False)] = 6
    numeric = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    out[(out < 0) & numeric.notna()] = numeric.clip(upper=6).fillna(-1).astype("int16")
    return out


def yes_flag(series: pd.Series) -> pd.Series:
    text = normalize_text(series)
    numeric = pd.to_numeric(text, errors="coerce")
    flag = numeric.eq(1)
    flag |= text.str.contains("1|Y|YES|TRUE|예|사용|해당", case=False, regex=True, na=False)
    return flag.astype("int8")


def make_treatment_tokens(df: pd.DataFrame) -> pd.DataFrame:
    treatment = text_series(df, TREATMENT_COL).str.upper()
    specific_raw = text_series(df, SPECIFIC_TREATMENT_COL)
    specific = specific_raw.str.upper()
    out = pd.DataFrame(index=df.index)

    out["is_ivf"] = treatment.str.contains("IVF", na=False).astype("int8")
    out["is_di"] = treatment.str.contains("DI", na=False).astype("int8")
    out["has_icsi"] = specific.str.contains("ICSI|미세", regex=True, na=False).astype("int8")
    out["has_ivf_token"] = specific.str.contains("IVF", na=False).astype("int8")
    out["has_iui"] = specific.str.contains("IUI", na=False).astype("int8")
    out["has_ici"] = specific.str.contains("ICI", na=False).astype("int8")
    out["has_fer"] = specific.str.contains("FER", na=False).astype("int8")
    out["has_blastocyst"] = specific.str.contains("BLAST|배반포", regex=True, na=False).astype("int8")
    out["has_ah"] = specific.str.contains(r"\bAH\b|보조", regex=True, na=False).astype("int8")
    out["has_unknown_specific_treatment"] = (
        specific.str.contains("UNKNOWN|알수|미상", regex=True, na=False) | specific.eq(MISSING_CATEGORY)
    ).astype("int8")

    split = specific.str.replace(":", "/", regex=False).str.replace(",", "/", regex=False).str.split("/")
    out["specific_treatment_token_count"] = split.apply(lambda xs: len([x.strip() for x in xs if x.strip() and x != MISSING_CATEGORY])).astype("int16")
    out["specific_treatment_has_slash"] = specific.str.contains("/", regex=False, na=False).astype("int8")
    out["specific_treatment_has_colon"] = specific.str.contains(":", regex=False, na=False).astype("int8")
    out["specific_treatment_raw_normalized"] = specific_raw

    pattern = pd.Series("RAW", index=df.index, dtype="object")
    pattern[specific.eq(MISSING_CATEGORY)] = "MISSING"
    pattern[specific.str.contains("/", regex=False, na=False)] = "SLASH_COMBO"
    pattern[specific.str.contains(":", regex=False, na=False)] = "COLON_COMBO"
    pattern[specific.str.contains("UNKNOWN|알수|미상", regex=True, na=False)] = "UNKNOWN_INCLUDED"
    pattern[(pattern == "RAW") & specific.eq(MISSING_CATEGORY)] = "MISSING"
    out["specific_treatment_pattern"] = pattern
    return out


def make_reason_tokens(df: pd.DataFrame) -> pd.DataFrame:
    reason = text_series(df, EMBRYO_REASON_COL)
    out = pd.DataFrame(index=df.index)
    out["reason_current_treatment"] = reason.str.contains("현재|시술|치료", regex=True, na=False).astype("int8")
    out["reason_embryo_storage"] = reason.str.contains("배아.*저장|저장.*배아", regex=True, na=False).astype("int8")
    out["reason_oocyte_storage"] = reason.str.contains("난자.*저장|저장.*난자", regex=True, na=False).astype("int8")
    out["reason_donation"] = reason.str.contains("기증", regex=True, na=False).astype("int8")
    out["reason_research"] = reason.str.contains("연구", regex=True, na=False).astype("int8")
    out["reason_missing"] = reason.eq(MISSING_CATEGORY).astype("int8")
    tokens = reason.str.replace(",", "/", regex=False).str.replace(":", "/", regex=False).str.split("/")
    out["reason_token_count"] = tokens.apply(lambda xs: len([x.strip() for x in xs if x.strip() and x != MISSING_CATEGORY])).astype("int16")
    out["storage_only_flag"] = (
        (out["reason_embryo_storage"].eq(1) | out["reason_oocyte_storage"].eq(1)) & out["reason_current_treatment"].eq(0)
    ).astype("int8")
    out["donation_only_flag"] = (out["reason_donation"].eq(1) & out["reason_current_treatment"].eq(0)).astype("int8")
    out["current_treatment_absent_flag"] = out["reason_current_treatment"].eq(0).astype("int8")
    out["reason_raw_normalized"] = reason

    branch = pd.Series("other", index=df.index, dtype="object")
    branch[out["reason_missing"].eq(1)] = "missing"
    branch[(out["reason_current_treatment"].eq(1)) & (out["reason_donation"].eq(0)) & (out["reason_embryo_storage"].eq(0)) & (out["reason_oocyte_storage"].eq(0))] = "current_treatment_only"
    branch[(out["reason_current_treatment"].eq(1)) & (out["reason_donation"].eq(1))] = "current_treatment_plus_donation"
    branch[(out["reason_current_treatment"].eq(1)) & ((out["reason_embryo_storage"].eq(1)) | (out["reason_oocyte_storage"].eq(1)))] = "current_treatment_plus_storage"
    branch[(out["reason_embryo_storage"].eq(1)) & (out["reason_current_treatment"].eq(0)) & (out["reason_donation"].eq(0))] = "embryo_storage_only"
    branch[(out["reason_oocyte_storage"].eq(1)) & (out["reason_current_treatment"].eq(0)) & (out["reason_donation"].eq(0))] = "oocyte_storage_only"
    branch[(out["reason_donation"].eq(1)) & (out["reason_current_treatment"].eq(0)) & (out["storage_only_flag"].eq(0))] = "donation_only"
    branch[(out["reason_donation"].eq(1)) & (out["storage_only_flag"].eq(1))] = "storage_plus_donation"
    out["reason_branch"] = branch
    return out


def make_art_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise ART cycle-level feature engineering.
    This function must not use target, train statistics, or test distribution.
    """
    features = df.drop(columns=[c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]).copy()

    for col in features.select_dtypes(include=["object", "category"]).columns:
        features[col] = normalize_text(features[col])

    age = text_series(df, AGE_COL)
    age_ord = map_age_ordinal(age)
    features["age_ord"] = age_ord
    features["age_unknown_flag"] = age_ord.eq(-1).astype("int8")
    features["age_group_raw"] = age
    age_names = {
        "age_is_18_34": "만18-34세",
        "age_is_35_37": "만35-37세",
        "age_is_38_39": "만38-39세",
        "age_is_40_42": "만40-42세",
        "age_is_43_44": "만43-44세",
        "age_is_45_50": "만45-50세",
    }
    for name, raw in age_names.items():
        features[name] = age.eq(raw).astype("int8")

    treatment_features = make_treatment_tokens(df)
    features = pd.concat([features, treatment_features], axis=1)

    egg_source = text_series(df, EGG_SOURCE_COL)
    sperm_source = text_series(df, SPERM_SOURCE_COL)
    features["is_own_egg"] = egg_source.str.contains("본인|자기|Own", case=False, regex=True, na=False).astype("int8")
    features["is_donor_egg"] = egg_source.str.contains("기증|Donor", case=False, regex=True, na=False).astype("int8")
    features["is_partner_sperm"] = sperm_source.str.contains("배우자|파트너|Partner", case=False, regex=True, na=False).astype("int8")
    features["is_donor_sperm"] = sperm_source.str.contains("기증|Donor", case=False, regex=True, na=False).astype("int8")
    features["is_donor_embryo"] = yes_flag(text_series(df, DONOR_EMBRYO_COL))
    features["is_surrogacy"] = yes_flag(text_series(df, SURROGACY_COL))
    features["egg_source_raw"] = egg_source
    features["sperm_source_raw"] = sperm_source
    features["egg_donor_age_raw"] = text_series(df, EGG_DONOR_AGE_COL)
    features["sperm_donor_age_raw"] = text_series(df, SPERM_DONOR_AGE_COL)
    features["age_x_donor_egg"] = age_ord * features["is_donor_egg"]
    features["age_x_own_egg"] = age_ord * features["is_own_egg"]
    features["age_x_donor_sperm"] = age_ord * features["is_donor_sperm"]

    features["is_fresh_embryo"] = yes_flag(text_series(df, FRESH_EMBRYO_COL))
    features["is_frozen_embryo"] = yes_flag(text_series(df, FROZEN_EMBRYO_COL))
    fresh = features["is_fresh_embryo"].eq(1)
    frozen = features["is_frozen_embryo"].eq(1)
    combo = pd.Series("neither", index=df.index, dtype="object")
    combo[fresh & ~frozen] = "fresh_only"
    combo[frozen & ~fresh] = "frozen_only"
    combo[fresh & frozen] = "both"
    combo[text_series(df, FRESH_EMBRYO_COL).eq(MISSING_CATEGORY) | text_series(df, FROZEN_EMBRYO_COL).eq(MISSING_CATEGORY)] = "missing_or_unknown"
    features["fresh_frozen_combo"] = combo
    embryo_thaw_count = numeric_series(df, EMBRYO_THAW_COUNT_COL)
    oocyte_thaw_count = numeric_series(df, OOCYTE_THAW_COUNT_COL)
    features["has_embryo_thaw"] = embryo_thaw_count.fillna(0).gt(0).astype("int8")
    features["has_oocyte_thaw"] = oocyte_thaw_count.fillna(0).gt(0).astype("int8")
    features["embryo_thaw_day_missing"] = numeric_series(df, EMBRYO_THAW_DAY_COL).isna().astype("int8")
    features["oocyte_thaw_day_missing"] = numeric_series(df, OOCYTE_THAW_DAY_COL).isna().astype("int8")
    features["embryo_thaw_count_positive"] = embryo_thaw_count.gt(0).astype("int8")
    features["oocyte_thaw_count_positive"] = oocyte_thaw_count.gt(0).astype("int8")

    transfer_count = numeric_series(df, TRANSFER_COUNT_COL)
    transfer_day = numeric_series(df, TRANSFER_DAY_COL)
    features["embryo_transferred_flag"] = transfer_count.gt(0).astype("int8")
    features["no_embryo_transfer_flag"] = transfer_count.fillna(0).eq(0).astype("int8")
    features["embryo_transfer_count_missing"] = transfer_count.isna().astype("int8")
    features["embryo_transfer_count_raw"] = transfer_count
    transfer_bin = pd.Series("missing", index=df.index, dtype="object")
    transfer_bin[transfer_count.eq(0)] = "0"
    transfer_bin[transfer_count.eq(1)] = "1"
    transfer_bin[transfer_count.eq(2)] = "2"
    transfer_bin[transfer_count.ge(3)] = "3+"
    features["embryo_transfer_count_bin"] = transfer_bin
    features["single_embryo_transfer_flag"] = yes_flag(text_series(df, SINGLE_TRANSFER_COL))
    features["multi_embryo_transfer_flag"] = transfer_count.gt(1).astype("int8")
    features["transfer_day_raw"] = transfer_day
    features["transfer_day_missing"] = transfer_day.isna().astype("int8")
    features["transfer_day_0_1"] = transfer_day.between(0, 1, inclusive="both").astype("int8")
    features["transfer_day_2_3"] = transfer_day.between(2, 3, inclusive="both").astype("int8")
    features["transfer_day_4_6"] = transfer_day.between(4, 6, inclusive="both").astype("int8")
    features["transfer_day_5"] = transfer_day.eq(5).astype("int8")
    features["transfer_day_ge5"] = transfer_day.ge(5).astype("int8")
    features["possible_blastocyst_transfer"] = ((features["transfer_day_ge5"].eq(1)) | (features["has_blastocyst"].eq(1))).astype("int8")
    features["transfer_day_x_transfer_count"] = transfer_day * transfer_count
    features["age_x_transfer_flag"] = age_ord * features["embryo_transferred_flag"]
    features["age_x_transfer_day5"] = age_ord * features["transfer_day_5"]

    reason_features = make_reason_tokens(df)
    features = pd.concat([features, reason_features], axis=1)

    for col in FUNNEL_COUNT_COLUMNS:
        if col not in df.columns:
            continue
        safe_name = make_safe_feature_name(col)
        s = numeric_series(df, col)
        features[f"{safe_name}_missing_flag"] = s.isna().astype("int8")
        features[f"{safe_name}_zero_flag"] = s.eq(0).astype("int8")
        features[f"{safe_name}_positive_flag"] = s.gt(0).astype("int8")
        features[f"{safe_name}_bin"] = bin_count(s)
        features[f"{safe_name}_log1p"] = np.log1p(s.where(s.ge(0)))

    features["embryo_creation_rate"] = safe_divide(numeric_series(df, "총 생성 배아 수"), numeric_series(df, "혼합된 난자 수"))
    features["icsi_embryo_creation_rate"] = safe_divide(numeric_series(df, "미세주입에서 생성된 배아 수"), numeric_series(df, "미세주입된 난자 수"))
    features["transfer_per_embryo_rate"] = safe_divide(numeric_series(df, TRANSFER_COUNT_COL), numeric_series(df, "총 생성 배아 수"))
    features["storage_per_embryo_rate"] = safe_divide(numeric_series(df, "저장된 배아 수"), numeric_series(df, "총 생성 배아 수"))
    features["partner_sperm_mix_share"] = safe_divide(numeric_series(df, "파트너 정자와 혼합된 난자 수"), numeric_series(df, "혼합된 난자 수"))
    features["donor_sperm_mix_share"] = safe_divide(numeric_series(df, "기증자 정자와 혼합된 난자 수"), numeric_series(df, "혼합된 난자 수"))
    features["thawed_embryo_transfer_proxy"] = safe_divide(numeric_series(df, TRANSFER_COUNT_COL), numeric_series(df, EMBRYO_THAW_COUNT_COL))
    features["transfer_gt_created_flag"] = (numeric_series(df, TRANSFER_COUNT_COL) > numeric_series(df, "총 생성 배아 수")).astype("int8")
    features["fresh_current_funnel_valid_flag"] = ((features["is_fresh_embryo"].eq(1)) & (features["reason_current_treatment"].eq(1))).astype("int8")
    features["frozen_funnel_flag"] = features["is_frozen_embryo"].astype("int8")
    features["icsi_funnel_flag"] = features["has_icsi"].astype("int8")

    for col, new_col in HISTORY_COLUMNS.items():
        features[new_col] = map_count_category(text_series(df, col))
    total_treatment = features.get("total_treatment_count_ord", pd.Series(-1, index=df.index))
    total_pregnancy = features.get("total_pregnancy_count_ord", pd.Series(-1, index=df.index))
    total_birth = features.get("total_birth_count_ord", pd.Series(-1, index=df.index))
    features["previous_pregnancy_flag"] = total_pregnancy.gt(0).astype("int8")
    features["previous_live_birth_flag"] = total_birth.gt(0).astype("int8")
    features["previous_ivf_pregnancy_flag"] = features.get("ivf_pregnancy_count_ord", pd.Series(-1, index=df.index)).gt(0).astype("int8")
    features["previous_ivf_birth_flag"] = features.get("ivf_birth_count_ord", pd.Series(-1, index=df.index)).gt(0).astype("int8")
    features["repeated_treatment_flag"] = total_treatment.gt(1).astype("int8")
    features["repeated_failure_proxy"] = ((total_treatment.gt(1)) & (total_pregnancy.le(0))).astype("int8")
    features["birth_per_treatment_proxy"] = safe_divide(total_birth, total_treatment)
    features["pregnancy_per_treatment_proxy"] = safe_divide(total_pregnancy, total_treatment)

    cause = make_cause_features(df)
    features = pd.concat([features, cause], axis=1)

    features = features.loc[:, ~features.columns.duplicated()]
    return features


def make_cause_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def any_positive(cols: list[str]) -> pd.Series:
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series(False, index=df.index)
        return df[existing].apply(lambda row: pd.to_numeric(row, errors="coerce").fillna(0).gt(0).any(), axis=1)

    male_cols = [c for c in CAUSE_COLUMNS if "남성" in c or "정자" in c]
    female_cols = [c for c in CAUSE_COLUMNS if "여성" in c or "난관" in c or "배란" in c or "자궁" in c or "자궁내막증" in c]
    couple_cols = [c for c in CAUSE_COLUMNS if "부부" in c]
    unexplained_cols = [c for c in CAUSE_COLUMNS if "불명확" in c]
    out["male_factor_any"] = any_positive(male_cols).astype("int8")
    out["female_factor_any"] = any_positive(female_cols).astype("int8")
    out["couple_factor_any"] = any_positive(couple_cols).astype("int8")
    out["unexplained_factor"] = any_positive(unexplained_cols).astype("int8")
    out["sperm_factor_count"] = sum((numeric_series(df, c).fillna(0).gt(0)).astype("int8") for c in male_cols if c in df.columns).astype("int16") if any(c in df.columns for c in male_cols) else 0
    out["female_factor_count"] = sum((numeric_series(df, c).fillna(0).gt(0)).astype("int8") for c in female_cols if c in df.columns).astype("int16") if any(c in df.columns for c in female_cols) else 0
    out["total_cause_count"] = out[["male_factor_any", "female_factor_any", "couple_factor_any", "unexplained_factor"]].sum(axis=1).astype("int16")
    out["both_male_female_factor"] = ((out["male_factor_any"].eq(1)) & (out["female_factor_any"].eq(1))).astype("int8")
    out["tubal_factor_flag"] = (numeric_series(df, "불임 원인 - 난관 질환").fillna(0).gt(0)).astype("int8")
    out["ovulatory_factor_flag"] = (numeric_series(df, "불임 원인 - 배란 장애").fillna(0).gt(0)).astype("int8")
    out["endometriosis_flag"] = (numeric_series(df, "불임 원인 - 자궁내막증").fillna(0).gt(0)).astype("int8")
    out["cervical_factor_flag"] = (numeric_series(df, "불임 원인 - 자궁경부 문제").fillna(0).gt(0)).astype("int8")
    out["sperm_concentration_flag"] = (numeric_series(df, "불임 원인 - 정자 농도").fillna(0).gt(0)).astype("int8")
    out["sperm_motility_flag"] = (numeric_series(df, "불임 원인 - 정자 운동성").fillna(0).gt(0)).astype("int8")
    out["sperm_morphology_flag"] = (numeric_series(df, "불임 원인 - 정자 형태").fillna(0).gt(0)).astype("int8")
    out["sperm_immunologic_flag"] = (numeric_series(df, "불임 원인 - 정자 면역학적 요인").fillna(0).gt(0)).astype("int8")
    return out


def make_safe_feature_name(col: str) -> str:
    mapping = {
        "수집된 신선 난자 수": "fresh_oocyte_collected_count",
        "혼합된 난자 수": "mixed_oocyte_count",
        "파트너 정자와 혼합된 난자 수": "partner_sperm_mixed_oocyte_count",
        "기증자 정자와 혼합된 난자 수": "donor_sperm_mixed_oocyte_count",
        "미세주입된 난자 수": "icsi_oocyte_count",
        "미세주입에서 생성된 배아 수": "icsi_embryo_created_count",
        "총 생성 배아 수": "embryo_created_count",
        "이식된 배아 수": "embryo_transferred_count",
        "미세주입 배아 이식 수": "icsi_embryo_transferred_count",
        "저장된 배아 수": "embryo_stored_count",
        "미세주입 후 저장된 배아 수": "icsi_embryo_stored_count",
        "해동된 배아 수": "embryo_thawed_count",
        "해동 난자 수": "oocyte_thawed_count",
    }
    return mapping.get(col, re.sub(r"\W+", "_", col).strip("_").lower())


def get_feature_families(feature_columns: list[str] | None = None) -> dict[str, list[str]]:
    families = {
        "raw_base": [],
        "branch_flags": [
            "is_ivf",
            "is_di",
            "is_own_egg",
            "is_donor_egg",
            "is_partner_sperm",
            "is_donor_sperm",
            "is_donor_embryo",
            "is_surrogacy",
            "is_fresh_embryo",
            "is_frozen_embryo",
            "fresh_frozen_combo",
            "has_embryo_thaw",
            "has_oocyte_thaw",
            "embryo_thaw_day_missing",
            "oocyte_thaw_day_missing",
            "embryo_thaw_count_positive",
            "oocyte_thaw_count_positive",
        ],
        "treatment_tokens": [
            "has_icsi",
            "has_ivf_token",
            "has_iui",
            "has_ici",
            "has_fer",
            "has_blastocyst",
            "has_ah",
            "has_unknown_specific_treatment",
            "specific_treatment_token_count",
            "specific_treatment_has_slash",
            "specific_treatment_has_colon",
            "specific_treatment_raw_normalized",
            "specific_treatment_pattern",
        ],
        "age_source_interactions": [
            "age_ord",
            "age_unknown_flag",
            "age_group_raw",
            "age_is_18_34",
            "age_is_35_37",
            "age_is_38_39",
            "age_is_40_42",
            "age_is_43_44",
            "age_is_45_50",
            "egg_source_raw",
            "sperm_source_raw",
            "egg_donor_age_raw",
            "sperm_donor_age_raw",
            "age_x_donor_egg",
            "age_x_own_egg",
            "age_x_donor_sperm",
            "age_x_transfer_flag",
            "age_x_transfer_day5",
        ],
        "transfer_features": [
            "embryo_transferred_flag",
            "no_embryo_transfer_flag",
            "embryo_transfer_count_missing",
            "embryo_transfer_count_raw",
            "embryo_transfer_count_bin",
            "single_embryo_transfer_flag",
            "multi_embryo_transfer_flag",
            "transfer_day_raw",
            "transfer_day_missing",
            "transfer_day_0_1",
            "transfer_day_2_3",
            "transfer_day_4_6",
            "transfer_day_5",
            "transfer_day_ge5",
            "possible_blastocyst_transfer",
            "transfer_day_x_transfer_count",
        ],
        "embryo_reason_features": [
            "reason_current_treatment",
            "reason_embryo_storage",
            "reason_oocyte_storage",
            "reason_donation",
            "reason_research",
            "reason_missing",
            "reason_token_count",
            "storage_only_flag",
            "donation_only_flag",
            "current_treatment_absent_flag",
            "reason_raw_normalized",
            "reason_branch",
        ],
        "funnel_count_bin_features": [],
        "funnel_ratio_features": [
            "embryo_creation_rate",
            "icsi_embryo_creation_rate",
            "transfer_per_embryo_rate",
            "storage_per_embryo_rate",
            "partner_sperm_mix_share",
            "donor_sperm_mix_share",
            "thawed_embryo_transfer_proxy",
            "transfer_gt_created_flag",
            "fresh_current_funnel_valid_flag",
            "frozen_funnel_flag",
            "icsi_funnel_flag",
        ],
        "history_features": list(HISTORY_COLUMNS.values())
        + [
            "previous_pregnancy_flag",
            "previous_live_birth_flag",
            "previous_ivf_pregnancy_flag",
            "previous_ivf_birth_flag",
            "repeated_treatment_flag",
            "repeated_failure_proxy",
            "birth_per_treatment_proxy",
            "pregnancy_per_treatment_proxy",
        ],
        "cause_features": [
            "male_factor_any",
            "female_factor_any",
            "couple_factor_any",
            "unexplained_factor",
            "sperm_factor_count",
            "female_factor_count",
            "total_cause_count",
            "both_male_female_factor",
            "tubal_factor_flag",
            "ovulatory_factor_flag",
            "endometriosis_flag",
            "cervical_factor_flag",
            "sperm_concentration_flag",
            "sperm_motility_flag",
            "sperm_morphology_flag",
            "sperm_immunologic_flag",
        ],
    }
    for col in FUNNEL_COUNT_COLUMNS:
        safe = make_safe_feature_name(col)
        families["funnel_count_bin_features"].extend(
            [
                f"{safe}_missing_flag",
                f"{safe}_zero_flag",
                f"{safe}_positive_flag",
                f"{safe}_bin",
                f"{safe}_log1p",
            ]
        )

    if feature_columns is not None:
        assigned = {col for cols in families.values() for col in cols}
        raw_base = [c for c in feature_columns if c not in assigned and c not in {ID_COLUMN, TARGET_COLUMN}]
        families["raw_base"] = raw_base
        feature_set = set(feature_columns)
        families = {name: [c for c in cols if c in feature_set] for name, cols in families.items()}
    return families


def save_feature_families(path: Path, feature_columns: list[str]) -> dict[str, list[str]]:
    families = get_feature_families(feature_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(families, ensure_ascii=False, indent=2), encoding="utf-8")
    return families
