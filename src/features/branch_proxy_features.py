from __future__ import annotations

import numpy as np
import pandas as pd


PATCH_GROUPS = {
    "frozen_proxy",
    "donor_egg_proxy",
    "transfer_positive_proxy",
    "day5_proxy",
    "oocyte_embryo_nonlinear_proxy",
    "low_probability_branch_proxy",
}


def _s(features: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce")
    return pd.Series(default, index=features.index, dtype="float64")


def _cat(features: pd.DataFrame, col: str, default: str = "__MISSING__") -> pd.Series:
    if col in features.columns:
        return features[col].fillna(default).astype(str)
    return pd.Series(default, index=features.index, dtype="object")


def _flag(features: pd.DataFrame, col: str) -> pd.Series:
    return _s(features, col).fillna(0).gt(0).astype("int8")


def _masked(mask: pd.Series, values: pd.Series, fill: float = 0.0) -> pd.Series:
    return values.where(mask.astype(bool), fill)


def _record(
    rows: list[dict[str, object]],
    feature_name: str,
    feature_group: str,
    created: bool,
    source_columns: list[str],
    dtype: str,
    missing_handling: str,
    reason: str,
) -> None:
    rows.append(
        {
            "feature_name": feature_name,
            "feature_group": feature_group,
            "created": created,
            "source_columns": ";".join(source_columns),
            "dtype": dtype,
            "missing_handling": missing_handling,
            "reason": reason,
        }
    )


def _add(
    out: pd.DataFrame,
    rows: list[dict[str, object]],
    name: str,
    group: str,
    value: pd.Series,
    source_columns: list[str],
    dtype: str,
    missing_handling: str,
    reason: str,
) -> None:
    out[name] = value
    _record(rows, name, group, True, source_columns, dtype, missing_handling, reason)


def add_branch_proxy_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-wise branch-proxy features after make_art_features().

    Principles:
    - row-wise transformation only
    - no target usage
    - no test statistics
    - no train/test concat
    """
    out = features.copy()
    metadata: list[dict[str, object]] = []

    age = _s(out, "age_ord")
    frozen = _flag(out, "is_frozen_embryo")
    donor_egg = _flag(out, "is_donor_egg")
    own_egg = _flag(out, "is_own_egg")
    partner_sperm = _flag(out, "is_partner_sperm")
    donor_sperm = _flag(out, "is_donor_sperm")
    fresh = _flag(out, "is_fresh_embryo")
    transfer_positive = _flag(out, "embryo_transferred_flag")
    day5 = _flag(out, "transfer_day_5")
    single_transfer = _flag(out, "single_embryo_transfer_flag")
    multi_transfer = _flag(out, "multi_embryo_transfer_flag")
    reason_storage = _flag(out, "reason_embryo_storage")
    reason_current = _flag(out, "reason_current_treatment")

    transfer_day = _s(out, "transfer_day_raw")
    transfer_day_6plus = transfer_day.ge(6).astype("int8")
    embryo_transferred = _s(out, "embryo_transfer_count_raw")
    embryo_created = _s(out, "총 생성 배아 수")
    embryo_stored = _s(out, "저장된 배아 수")
    embryo_thawed = _s(out, "해동된 배아 수")
    fresh_oocyte = _s(out, "수집된 신선 난자 수")

    storage_rate = _s(out, "storage_per_embryo_rate").fillna(0)
    transfer_rate = _s(out, "transfer_per_embryo_rate").fillna(0)
    thaw_transfer_proxy = _s(out, "thawed_embryo_transfer_proxy").fillna(0)
    embryo_creation_rate = _s(out, "embryo_creation_rate").fillna(0)

    # 4.1 Frozen branch proxy
    frozen_sources = ["is_frozen_embryo"]
    for name, values, sources in [
        ("frozen_x_age_ord", age, ["age_ord"]),
        ("frozen_x_own_egg", own_egg, ["is_own_egg"]),
        ("frozen_x_donor_egg", donor_egg, ["is_donor_egg"]),
        ("frozen_x_transfer_day_5", day5, ["transfer_day_5"]),
        ("frozen_x_transfer_day_6plus", transfer_day_6plus, ["transfer_day_raw"]),
        ("frozen_x_single_embryo_transfer", single_transfer, ["single_embryo_transfer_flag"]),
        ("frozen_x_multi_embryo_transfer", multi_transfer, ["multi_embryo_transfer_flag"]),
        ("frozen_x_embryo_transferred_count", embryo_transferred.fillna(0), ["embryo_transfer_count_raw"]),
        ("frozen_x_embryo_stored_count", embryo_stored.fillna(0), ["저장된 배아 수"]),
        ("frozen_x_embryo_thawed_count", embryo_thawed.fillna(0), ["해동된 배아 수"]),
        ("frozen_x_storage_per_embryo_rate", storage_rate, ["storage_per_embryo_rate"]),
        ("frozen_x_transfer_per_embryo_rate", transfer_rate, ["transfer_per_embryo_rate"]),
        ("frozen_x_thawed_embryo_transfer_proxy", thaw_transfer_proxy, ["thawed_embryo_transfer_proxy"]),
        ("frozen_x_reason_embryo_storage", reason_storage, ["reason_embryo_storage"]),
        ("frozen_x_reason_current_treatment", reason_current, ["reason_current_treatment"]),
    ]:
        _add(out, metadata, name, "frozen_proxy", _masked(frozen, values), frozen_sources + sources, "numeric", "NaN filled to 0 inside frozen interaction", "Frozen branch may have different embryo age/storage dynamics.")
    for name, values, sources in [
        ("frozen_embryo_thawed_missing_flag", embryo_thawed.isna().astype("int8"), ["해동된 배아 수"]),
        ("frozen_transfer_day_missing_flag", transfer_day.isna().astype("int8"), ["transfer_day_raw"]),
        ("frozen_embryo_count_missing_flag", embryo_created.isna().astype("int8"), ["총 생성 배아 수"]),
    ]:
        _add(out, metadata, name, "frozen_proxy", _masked(frozen, values), frozen_sources + sources, "int8", "Missing expressed as branch-gated flag", "Frozen branch structural missingness monitor.")

    # 4.2 Donor egg branch proxy
    donor_sources = ["is_donor_egg"]
    for name, values, sources in [
        ("donor_egg_x_recipient_age_ord", age, ["age_ord"]),
        ("donor_egg_x_partner_sperm", partner_sperm, ["is_partner_sperm"]),
        ("donor_egg_x_donor_sperm", donor_sperm, ["is_donor_sperm"]),
        ("donor_egg_x_transfer_day_5", day5, ["transfer_day_5"]),
        ("donor_egg_x_transfer_day_6plus", transfer_day_6plus, ["transfer_day_raw"]),
        ("donor_egg_x_single_embryo_transfer", single_transfer, ["single_embryo_transfer_flag"]),
        ("donor_egg_x_multi_embryo_transfer", multi_transfer, ["multi_embryo_transfer_flag"]),
        ("donor_egg_x_fresh", fresh, ["is_fresh_embryo"]),
        ("donor_egg_x_frozen", frozen, ["is_frozen_embryo"]),
        ("donor_egg_x_embryo_transferred_count", embryo_transferred.fillna(0), ["embryo_transfer_count_raw"]),
        ("donor_egg_x_embryo_created_count", embryo_created.fillna(0), ["총 생성 배아 수"]),
        ("donor_egg_x_embryo_stored_count", embryo_stored.fillna(0), ["저장된 배아 수"]),
        ("donor_egg_x_storage_per_embryo_rate", storage_rate, ["storage_per_embryo_rate"]),
    ]:
        _add(out, metadata, name, "donor_egg_proxy", _masked(donor_egg, values), donor_sources + sources, "numeric", "NaN filled to 0 inside donor-egg interaction", "Donor egg branch separates recipient age from oocyte age.")

    # 4.3 Transfer-positive branch proxy
    tp_sources = ["embryo_transferred_flag"]
    for name, values, sources in [
        ("transfer_positive_x_age_ord", age, ["age_ord"]),
        ("transfer_positive_x_own_egg", own_egg, ["is_own_egg"]),
        ("transfer_positive_x_donor_egg", donor_egg, ["is_donor_egg"]),
        ("transfer_positive_x_fresh", fresh, ["is_fresh_embryo"]),
        ("transfer_positive_x_frozen", frozen, ["is_frozen_embryo"]),
        ("transfer_positive_x_transfer_day_5", day5, ["transfer_day_5"]),
        ("transfer_positive_x_transfer_day_6plus", transfer_day_6plus, ["transfer_day_raw"]),
        ("transfer_positive_x_single_embryo_transfer", single_transfer, ["single_embryo_transfer_flag"]),
        ("transfer_positive_x_multi_embryo_transfer", multi_transfer, ["multi_embryo_transfer_flag"]),
        ("transfer_positive_x_embryo_created_count", embryo_created.fillna(0), ["총 생성 배아 수"]),
        ("transfer_positive_x_embryo_stored_count", embryo_stored.fillna(0), ["저장된 배아 수"]),
        ("transfer_positive_x_storage_per_embryo_rate", storage_rate, ["storage_per_embryo_rate"]),
        ("transfer_positive_x_transfer_per_embryo_rate", transfer_rate, ["transfer_per_embryo_rate"]),
        ("transfer_positive_x_embryo_creation_rate", embryo_creation_rate, ["embryo_creation_rate"]),
    ]:
        _add(out, metadata, name, "transfer_positive_proxy", _masked(transfer_positive, values), tp_sources + sources, "numeric", "NaN filled to 0 inside transfer-positive interaction", "Focus on variance after transfer occurred.")

    age_group = _cat(out, "age_group_raw")
    for name, value in [
        ("transfer_positive_age_bin_x_day5", age_group.where(transfer_positive.eq(1) & day5.eq(1), "not_applicable")),
        ("transfer_positive_age_bin_x_frozen", age_group.where(transfer_positive.eq(1) & frozen.eq(1), "not_applicable")),
        ("transfer_positive_age_bin_x_donor_egg", age_group.where(transfer_positive.eq(1) & donor_egg.eq(1), "not_applicable")),
    ]:
        _add(out, metadata, name, "transfer_positive_proxy", value, ["age_group_raw", "embryo_transferred_flag"], "category", "Non-branch rows set to not_applicable", "Categorical interaction for CatBoost branch structure.")

    # 4.4 Day5 / blastocyst proxy
    blast = _flag(out, "has_blastocyst")
    for name, values, sources in [
        ("transfer_day_6plus", transfer_day_6plus, ["transfer_day_raw"]),
        ("explicit_blastocyst_token", blast, ["has_blastocyst"]),
        ("day5_explicit_blastocyst", (day5.eq(1) & blast.eq(1)).astype("int8"), ["transfer_day_5", "has_blastocyst"]),
        ("day5_without_explicit_blastocyst_token", (day5.eq(1) & blast.eq(0)).astype("int8"), ["transfer_day_5", "has_blastocyst"]),
        ("blastocyst_token_x_transfer_day_5", (blast.eq(1) & day5.eq(1)).astype("int8"), ["has_blastocyst", "transfer_day_5"]),
        ("blastocyst_token_x_transfer_day_6plus", (blast.eq(1) & transfer_day_6plus.eq(1)).astype("int8"), ["has_blastocyst", "transfer_day_raw"]),
        ("single_embryo_x_day5", (single_transfer.eq(1) & day5.eq(1)).astype("int8"), ["single_embryo_transfer_flag", "transfer_day_5"]),
        ("multi_embryo_x_day5", (multi_transfer.eq(1) & day5.eq(1)).astype("int8"), ["multi_embryo_transfer_flag", "transfer_day_5"]),
        ("fresh_x_day5", (fresh.eq(1) & day5.eq(1)).astype("int8"), ["is_fresh_embryo", "transfer_day_5"]),
        ("frozen_x_day5", (frozen.eq(1) & day5.eq(1)).astype("int8"), ["is_frozen_embryo", "transfer_day_5"]),
        ("donor_egg_x_day5", (donor_egg.eq(1) & day5.eq(1)).astype("int8"), ["is_donor_egg", "transfer_day_5"]),
        ("own_egg_x_day5", (own_egg.eq(1) & day5.eq(1)).astype("int8"), ["is_own_egg", "transfer_day_5"]),
    ]:
        _add(out, metadata, name, "day5_proxy", values, sources, "int8", "Missing transfer day implies 0 for boolean proxy", "Limited blastocyst/day5 morphology proxy.")

    # 4.5 Oocyte / embryo nonlinear proxy
    fresh_oocyte_11_20 = fresh_oocyte.between(11, 20, inclusive="both").astype("int8")
    embryo_6_20 = embryo_created.between(6, 20, inclusive="both").astype("int8")
    for name, values, sources, dtype in [
        ("fresh_oocyte_11_20_optimal", fresh_oocyte_11_20, ["수집된 신선 난자 수"], "int8"),
        ("fresh_oocyte_20plus_excess", fresh_oocyte.gt(20).astype("int8"), ["수집된 신선 난자 수"], "int8"),
        ("fresh_oocyte_distance_from_15", (fresh_oocyte - 15).abs().fillna(0), ["수집된 신선 난자 수"], "numeric"),
        ("embryo_created_6_20_window", embryo_6_20, ["총 생성 배아 수"], "int8"),
        ("embryo_created_20plus_excess", embryo_created.gt(20).astype("int8"), ["총 생성 배아 수"], "int8"),
        ("own_egg_x_fresh_oocyte_11_20_optimal", own_egg * fresh_oocyte_11_20, ["is_own_egg", "수집된 신선 난자 수"], "int8"),
        ("age_x_fresh_oocyte_11_20_optimal", age * fresh_oocyte_11_20, ["age_ord", "수집된 신선 난자 수"], "numeric"),
        ("fresh_current_x_fresh_oocyte_11_20_optimal", fresh * reason_current * fresh_oocyte_11_20, ["is_fresh_embryo", "reason_current_treatment", "수집된 신선 난자 수"], "int8"),
    ]:
        _add(out, metadata, name, "oocyte_embryo_nonlinear_proxy", values, sources, dtype, "Missing counts mapped to 0 for proxy/distance", "Nonlinear oocyte/embryo yield window proxy.")

    # 4.6 Low-probability structural branch monitor
    storage_only = _flag(out, "storage_only_flag")
    donation_only = _flag(out, "donation_only_flag")
    age_unknown = _flag(out, "age_unknown_flag")
    no_transfer = _flag(out, "no_embryo_transfer_flag")
    no_current = _flag(out, "current_treatment_absent_flag")
    for name, values, sources in [
        ("storage_only_low_prob_proxy", storage_only, ["storage_only_flag"]),
        ("donation_only_low_prob_proxy", donation_only, ["donation_only_flag"]),
        ("age_unknown_low_prob_proxy", age_unknown, ["age_unknown_flag"]),
        ("no_current_treatment_no_transfer_proxy", (no_current.eq(1) & no_transfer.eq(1)).astype("int8"), ["current_treatment_absent_flag", "no_embryo_transfer_flag"]),
        ("storage_or_donation_no_transfer_proxy", ((storage_only.eq(1) | donation_only.eq(1)) & no_transfer.eq(1)).astype("int8"), ["storage_only_flag", "donation_only_flag", "no_embryo_transfer_flag"]),
    ]:
        _add(out, metadata, name, "low_probability_branch_proxy", values, sources, "int8", "Boolean branch monitor", "Low-probability structural branch feature, not post-processing.")

    out.attrs["patch_feature_metadata"] = pd.DataFrame(metadata)
    return out


def get_patch_feature_metadata(features_with_patch: pd.DataFrame) -> pd.DataFrame:
    metadata = features_with_patch.attrs.get("patch_feature_metadata")
    if isinstance(metadata, pd.DataFrame):
        return metadata.copy()
    return pd.DataFrame(
        columns=[
            "feature_name",
            "feature_group",
            "created",
            "source_columns",
            "dtype",
            "missing_handling",
            "reason",
        ]
    )


def patch_feature_names(features_with_patch: pd.DataFrame) -> list[str]:
    metadata = get_patch_feature_metadata(features_with_patch)
    if metadata.empty:
        return []
    return metadata.loc[metadata["created"].eq(True), "feature_name"].tolist()
