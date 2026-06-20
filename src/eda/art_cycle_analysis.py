from __future__ import annotations

import html
import platform
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "eda"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
DICT_PATH = RAW_DIR / "데이터 명세.xlsx"

LEAKAGE_NOTE = (
    "This EDA uses train.csv only. test.csv was used only for schema validation, "
    "not for distributional analysis or feature design."
)

TARGET_CANDIDATES = ["임신 성공 여부", "pregnancy", "target"]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH) if TEST_PATH.exists() else None
    return train, test


def validate_schema_only(train: pd.DataFrame, test: pd.DataFrame | None) -> dict[str, object]:
    if test is None:
        return {"test_exists": False}
    train_features = [c for c in train.columns if c not in TARGET_CANDIDATES]
    test_columns = list(test.columns)
    return {
        "test_exists": True,
        "train_column_count": len(train.columns),
        "test_column_count": len(test.columns),
        "feature_schema_matches": train_features == test_columns,
        "train_only_columns": sorted(set(train_features) - set(test_columns)),
        "test_only_columns": sorted(set(test_columns) - set(train_features)),
        "test_shape_schema_only": test.shape,
    }


def detect_target(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    binary_cols = [
        c
        for c in df.columns
        if df[c].dropna().isin([0, 1]).all() and df[c].nunique(dropna=True) == 2
    ]
    if not binary_cols:
        raise ValueError("Target column could not be detected.")
    return binary_cols[-1]


def as_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def binary_from_text(series: pd.Series, yes_values: Iterable[str] = ("1", "Y", "Yes", "True", "예")) -> pd.Series:
    text = as_text(series).str.strip()
    return text.isin(set(yes_values)) | text.str.contains("사용|해당|예|Y|True|1", case=False, regex=True)


def make_branch_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    treatment = as_text(df.get("시술 유형", pd.Series(index=df.index, dtype=object))).str.upper()
    specific = as_text(df.get("특정 시술 유형", pd.Series(index=df.index, dtype=object))).str.upper()

    flags["is_ivf"] = treatment.str.contains("IVF", na=False)
    flags["is_di"] = treatment.str.contains("DI", na=False)
    flags["has_icsi"] = specific.str.contains("ICSI|미세", regex=True, na=False)
    flags["has_ivf_token"] = specific.str.contains("IVF", na=False)
    flags["has_iui"] = specific.str.contains("IUI", na=False)
    flags["has_ici"] = specific.str.contains("ICI", na=False)
    flags["has_fer"] = specific.str.contains("FER", na=False)
    flags["has_blastocyst"] = specific.str.contains("BLAST|배반포", regex=True, na=False)
    flags["has_ah"] = specific.str.contains(r"\bAH\b|보조", regex=True, na=False)
    flags["has_unknown_specific_treatment"] = specific.str.contains("UNKNOWN|알수|미상", regex=True, na=False) | specific.eq("")
    tokenized = specific.str.replace(":", "/", regex=False).str.replace(",", "/", regex=False)
    tokens = tokenized.str.split("/").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    flags["specific_treatment_token_count"] = tokens.apply(len)
    flags["specific_treatment_has_slash"] = specific.str.contains("/", regex=False, na=False)
    flags["specific_treatment_has_colon"] = specific.str.contains(":", regex=False, na=False)

    egg_source = as_text(df.get("난자 출처", pd.Series(index=df.index, dtype=object)))
    sperm_source = as_text(df.get("정자 출처", pd.Series(index=df.index, dtype=object)))
    donor_embryo = df.get("기증 배아 사용 여부", pd.Series(index=df.index, dtype=object))
    surrogacy = df.get("대리모 여부", pd.Series(index=df.index, dtype=object))

    flags["is_own_egg"] = egg_source.str.contains("본인|자기|Own", case=False, regex=True, na=False)
    flags["is_donor_egg"] = egg_source.str.contains("기증|Donor", case=False, regex=True, na=False)
    flags["is_partner_sperm"] = sperm_source.str.contains("배우자|파트너|Partner", case=False, regex=True, na=False)
    flags["is_donor_sperm"] = sperm_source.str.contains("기증|Donor", case=False, regex=True, na=False)
    flags["is_donor_embryo"] = binary_from_text(donor_embryo)
    flags["is_surrogacy"] = binary_from_text(surrogacy)

    flags["is_frozen_embryo"] = binary_from_text(df.get("동결 배아 사용 여부", pd.Series(index=df.index, dtype=object)))
    flags["is_fresh_embryo"] = binary_from_text(df.get("신선 배아 사용 여부", pd.Series(index=df.index, dtype=object)))
    flags["has_embryo_thaw"] = pd.to_numeric(df.get("해동된 배아 수", pd.Series(index=df.index)), errors="coerce").fillna(0).gt(0)
    flags["has_oocyte_thaw"] = pd.to_numeric(df.get("해동 난자 수", pd.Series(index=df.index)), errors="coerce").fillna(0).gt(0)

    transfer_count = pd.to_numeric(df.get("이식된 배아 수", pd.Series(index=df.index)), errors="coerce")
    transfer_day = pd.to_numeric(df.get("배아 이식 경과일", pd.Series(index=df.index)), errors="coerce")
    flags["embryo_transferred_flag"] = transfer_count.gt(0)
    flags["no_embryo_transfer_flag"] = transfer_count.fillna(0).eq(0)
    flags["embryo_transfer_count_missing"] = transfer_count.isna()
    flags["transfer_day_missing"] = transfer_day.isna()
    flags["transfer_day_0_1"] = transfer_day.between(0, 1, inclusive="both")
    flags["transfer_day_2_3"] = transfer_day.between(2, 3, inclusive="both")
    flags["transfer_day_4_6"] = transfer_day.between(4, 6, inclusive="both")
    flags["transfer_day_5"] = transfer_day.eq(5)
    flags["possible_blastocyst_transfer"] = transfer_day.ge(5) | flags["has_blastocyst"]

    reason = as_text(df.get("배아 생성 주요 이유", pd.Series(index=df.index, dtype=object)))
    flags["reason_current_treatment"] = reason.str.contains("현재|시술|치료", regex=True, na=False)
    flags["reason_embryo_storage"] = reason.str.contains("배아.*저장|저장.*배아", regex=True, na=False)
    flags["reason_oocyte_storage"] = reason.str.contains("난자.*저장|저장.*난자", regex=True, na=False)
    flags["reason_donation"] = reason.str.contains("기증", regex=True, na=False)
    flags["reason_research"] = reason.str.contains("연구", regex=True, na=False)
    reason_tokens = reason.str.replace(",", "/", regex=False).str.replace(":", "/", regex=False).str.split("/")
    flags["reason_token_count"] = reason_tokens.apply(lambda xs: len([x.strip() for x in xs if x.strip()]))
    flags["storage_only_flag"] = (flags["reason_embryo_storage"] | flags["reason_oocyte_storage"]) & ~flags["reason_current_treatment"]
    flags["current_treatment_absent_flag"] = ~flags["reason_current_treatment"]

    return flags


def summarize_target_rate(df: pd.DataFrame, by: str | list[str], target: str, min_count: int = 1) -> pd.DataFrame:
    cols = [by] if isinstance(by, str) else by
    if any(c not in df.columns for c in cols):
        return pd.DataFrame(columns=cols + ["n", "target_rate"])
    out = (
        df.groupby(cols, dropna=False)[target]
        .agg(n="size", target_rate="mean")
        .reset_index()
        .sort_values(["n"], ascending=False)
    )
    return out[out["n"] >= min_count]


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_rate": df.isna().mean().values,
                "dtype": [str(df[c].dtype) for c in df.columns],
            }
        )
        .sort_values(["missing_rate", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )


def save_table(name: str, df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    clean = df.copy()
    clean.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    tables[name] = clean


def numeric_summary(df: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "count": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p1": s.quantile(0.01),
                "p5": s.quantile(0.05),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "zero_rate": s.eq(0).mean(),
                "positive_rate": s.gt(0).mean(),
                "target_rate_when_zero": df.loc[s.eq(0), target].mean(),
                "target_rate_when_positive": df.loc[s.gt(0), target].mean(),
                "target_rate_when_missing": df.loc[s.isna(), target].mean(),
            }
        )
    return pd.DataFrame(rows)


def bin_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    labels = pd.Series("missing", index=s.index, dtype=object)
    labels[x.eq(0)] = "0"
    labels[x.eq(1)] = "1"
    labels[x.between(2, 3, inclusive="both")] = "2-3"
    labels[x.between(4, 5, inclusive="both")] = "4-5"
    labels[x.between(6, 10, inclusive="both")] = "6-10"
    labels[x.between(11, 15, inclusive="both")] = "11-15"
    labels[x.between(16, 20, inclusive="both")] = "16-20"
    labels[x.ge(21)] = "21+"
    return labels


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.gt(0)))


def make_funnel_features_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    pairs = {
        "embryo_creation_rate": ("총 생성 배아 수", "혼합된 난자 수"),
        "icsi_embryo_creation_rate": ("미세주입에서 생성된 배아 수", "미세주입된 난자 수"),
        "transfer_per_embryo_rate": ("이식된 배아 수", "총 생성 배아 수"),
        "storage_per_embryo_rate": ("저장된 배아 수", "총 생성 배아 수"),
        "partner_sperm_mix_share": ("파트너 정자와 혼합된 난자 수", "혼합된 난자 수"),
        "donor_sperm_mix_share": ("기증자 정자와 혼합된 난자 수", "혼합된 난자 수"),
        "thawed_embryo_transfer_proxy": ("이식된 배아 수", "해동된 배아 수"),
    }
    for name, (num, den) in pairs.items():
        if num in df.columns and den in df.columns:
            f[name] = ratio(df[num], df[den])
    return f


def tokenize_specific(series: pd.Series) -> pd.Series:
    text = as_text(series).str.upper()
    token_map = {
        "ICSI": r"ICSI|미세",
        "IVF": r"IVF",
        "IUI": r"IUI",
        "ICI": r"ICI",
        "FER": r"FER",
        "BLASTOCYST": r"BLAST|배반포",
        "AH": r"\bAH\b|보조",
        "Generic DI": r"\bDI\b",
        "Unknown": r"UNKNOWN|알수|미상|^$",
    }
    hits = pd.DataFrame({name: text.str.contains(pat, regex=True, na=False) for name, pat in token_map.items()}, index=series.index)
    return hits.apply(lambda row: "+".join(row.index[row].tolist()) if row.any() else "Other", axis=1)


def tokenize_reason(series: pd.Series) -> pd.Series:
    text = as_text(series)
    token_map = {
        "current_treatment": r"현재|시술|치료",
        "embryo_storage": r"배아.*저장|저장.*배아",
        "oocyte_storage": r"난자.*저장|저장.*난자",
        "donation": r"기증",
        "research": r"연구",
    }
    hits = pd.DataFrame({name: text.str.contains(pat, regex=True, na=False) for name, pat in token_map.items()}, index=series.index)
    labels = hits.apply(lambda row: "+".join(row.index[row].tolist()) if row.any() else "other", axis=1)
    labels[text.str.strip().eq("")] = "missing"
    return labels


def make_overview(df: pd.DataFrame, target: str, schema: dict[str, object]) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    id_dup = df["ID"].duplicated().sum() if "ID" in df.columns else np.nan
    rows = [
        ("rows", len(df)),
        ("columns", len(df.columns)),
        ("target_column", target),
        ("target_mean", df[target].mean()),
        ("target_zero_rate", (df[target] == 0).mean()),
        ("target_one_rate", (df[target] == 1).mean()),
        ("numeric_columns", len(numeric_cols)),
        ("categorical_columns", len(cat_cols)),
        ("id_column_exists", "ID" in df.columns),
        ("id_duplicate_count", id_dup),
        ("full_duplicate_rows", df.duplicated().sum()),
        ("schema_validation_only", schema.get("feature_schema_matches")),
        ("leakage_note", LEAKAGE_NOTE),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def branch_missingness(df: pd.DataFrame, flags: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    branch_masks = {
        "overall": pd.Series(True, index=df.index),
        "IVF": flags["is_ivf"],
        "DI": flags["is_di"],
        "transfer_count_0": pd.to_numeric(df.get("이식된 배아 수", pd.Series(index=df.index)), errors="coerce").fillna(0).eq(0),
        "transfer_count_positive": pd.to_numeric(df.get("이식된 배아 수", pd.Series(index=df.index)), errors="coerce").gt(0),
        "transfer_count_missing": pd.to_numeric(df.get("이식된 배아 수", pd.Series(index=df.index)), errors="coerce").isna(),
        "fresh_embryo": flags["is_fresh_embryo"],
        "frozen_embryo": flags["is_frozen_embryo"],
        "donor_egg": flags["is_donor_egg"],
        "own_egg": flags["is_own_egg"],
        "current_treatment_reason": flags["reason_current_treatment"],
        "storage_only": flags["storage_only_flag"],
        "donation_only": flags["reason_donation"] & ~flags["reason_current_treatment"],
    }
    rows = []
    for branch, mask in branch_masks.items():
        sub = df.loc[mask.fillna(False), columns]
        for col in columns:
            rows.append(
                {
                    "branch": branch,
                    "column": col,
                    "n": len(sub),
                    "missing_rate": sub[col].isna().mean() if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_missingness_interpretation(df: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    notes = {
        "배아 이식 경과일": ("yes", "transfer cycle", "Missing likely marks no-transfer or non-transfer branch.", "Use missing flag plus day bins."),
        "배아 해동 경과일": ("yes", "frozen/thaw", "Missing is expected outside frozen/thaw cycles.", "Condition interpretation on frozen branch."),
        "난자 해동 경과일": ("yes", "oocyte thaw", "Missing is expected outside oocyte thaw cycles.", "Use thaw flag and missing indicator."),
        "PGD 시술 여부": ("possible", "PGT", "Missing may mean not performed or not recorded.", "Keep explicit missing category."),
        "PGS 시술 여부": ("possible", "PGT", "Missing may mean not performed or not recorded.", "Keep explicit missing category."),
    }
    rows = []
    for col in df.columns:
        rate = float(missing.loc[missing["column"].eq(col), "missing_rate"].iloc[0])
        likely, branch, interp, handling = notes.get(
            col,
            (
                "unknown" if rate > 0 else "no",
                "all",
                "Review with branch tables before imputing.",
                "Use model-safe imputation after train-only fitting.",
            ),
        )
        rows.append(
            {
                "column": col,
                "overall_missing_rate": rate,
                "likely_structural_missing": likely,
                "related_branch": branch,
                "interpretation": interp,
                "feature_handling_note": handling,
            }
        )
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, top: int = 20) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    data = df.head(top).copy()
    labels = data[x].fillna("missing").map(str).tolist()
    values = pd.to_numeric(data[y], errors="coerce").fillna(0).to_numpy()
    plt.figure(figsize=(10, max(4, min(9, 0.35 * len(data)))))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap(pivot: pd.DataFrame, title: str, path: Path) -> None:
    if pivot.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(label="target rate")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_excel_report(tables: dict[str, pd.DataFrame]) -> None:
    preferred = [
        "overview",
        "missingness_overall",
        "unique_counts",
        "target_rate_by_age",
        "age_unknown_profile",
        "target_rate_by_age_egg_source",
        "target_rate_by_treatment_type",
        "target_rate_by_specific_treatment_token",
        "target_rate_by_embryo_reason_token",
        "missingness_by_branch",
        "missingness_interpretation",
        "funnel_numeric_summary",
        "funnel_target_rate_bins",
        "funnel_ratio_summary",
        "funnel_inconsistency_profile",
        "target_rate_by_embryo_transfer_count",
        "target_rate_by_transfer_day",
        "history_target_rate",
        "infertility_cause_target_rate",
        "infertility_cause_group_target_rate",
    ]
    xlsx_path = OUT_DIR / "eda_tables.xlsx"
    ordered = [(name, tables[name]) for name in preferred + [n for n in tables if n not in preferred] if name in tables]
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            used = set()
            for name, table in ordered:
                sheet = make_sheet_name(name, used)
                table.head(100000).to_excel(writer, sheet_name=sheet, index=False)
        return
    except ModuleNotFoundError:
        write_minimal_xlsx(xlsx_path, ordered)


def make_sheet_name(name: str, used: set[str]) -> str:
    sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31] or "sheet"
    base = sheet
    i = 1
    while sheet in used:
        suffix = str(i)
        sheet = f"{base[:31-len(suffix)]}{suffix}"
        i += 1
    used.add(sheet)
    return sheet


def excel_col_name(idx: int) -> str:
    name = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row_idx: int, col_idx: int, value: object) -> str:
    ref = f"{excel_col_name(col_idx)}{row_idx}"
    if pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating, bool)) and np.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value)}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def worksheet_xml(df: pd.DataFrame) -> str:
    limited = df.head(100000).copy()
    rows = []
    header = "".join(cell_xml(1, j, col) for j, col in enumerate(limited.columns))
    rows.append(f'<row r="1">{header}</row>')
    for i, (_, row) in enumerate(limited.iterrows(), start=2):
        cells = "".join(cell_xml(i, j, row[col]) for j, col in enumerate(limited.columns))
        rows.append(f'<row r="{i}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData></worksheet>'
    )


def write_minimal_xlsx(path: Path, ordered_tables: list[tuple[str, pd.DataFrame]]) -> None:
    used: set[str] = set()
    sheets = [(make_sheet_name(name, used), table) for name, table in ordered_tables]
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, (name, _) in enumerate(sheets, start=1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1)
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheets) + 1)
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f"{overrides}</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}</Relationships>",
        )
        for i, (_, table) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", worksheet_xml(table))


def make_feature_hypothesis_table() -> pd.DataFrame:
    items = [
        ("age_ord", "시술 당시 나이", "Age is a core ART baseline risk driver.", "Check age target-rate table.", "Donor egg can confound older groups.", "Ordinal encode plus unknown flag.", "high"),
        ("age_unknown_flag", "시술 당시 나이", "Unknown age may represent a distinct registry profile.", "Profile unknown age separately.", "Do not drop unknown group.", "Missing/unknown category flag.", "high"),
        ("is_donor_egg", "난자 출처", "Donor eggs can alter age-risk relationship.", "Age x egg source table.", "Strong confounding with age.", "Binary flag and interaction.", "high"),
        ("age_x_egg_source", "시술 당시 나이; 난자 출처", "Age effect differs by egg source.", "Cross-tab target rates.", "Small cells need smoothing.", "Interaction or grouped target encoding in CV only later.", "high"),
        ("is_donor_sperm", "정자 출처", "Sperm source may imply treatment branch.", "Sperm source target table.", "Confounded with DI.", "Binary flag.", "medium"),
        ("is_ivf", "시술 유형", "IVF and DI are different branches.", "Treatment branch target table.", "Branch-specific missingness.", "Binary branch flag.", "high"),
        ("is_di", "시술 유형", "DI has different lab/transfer fields.", "Treatment branch target table.", "Many embryo fields structurally absent.", "Binary branch flag.", "high"),
        ("has_icsi", "특정 시술 유형", "ICSI indicates male-factor/lab pathway.", "Specific treatment token table.", "Confounded with sperm factors.", "Token flag.", "medium"),
        ("has_blastocyst", "특정 시술 유형", "Blastocyst transfer proxy may matter.", "Token and day-5 tables.", "May be post-lab information.", "Flag with caution.", "medium"),
        ("is_fresh_embryo", "신선 배아 사용 여부", "Fresh and frozen cycles differ.", "Fresh/frozen target rates.", "Cycle timing confounding.", "Binary flag.", "high"),
        ("is_frozen_embryo", "동결 배아 사용 여부", "Frozen cycles differ from fresh.", "Fresh/frozen target rates.", "Thaw fields structural.", "Binary flag.", "high"),
        ("embryo_transferred_flag", "이식된 배아 수", "No-transfer cycles cannot have live pregnancy success in same way.", "Transfer count table.", "Post-cycle signal.", "Use explicitly for model scope decisions.", "high"),
        ("no_embryo_transfer_flag", "이식된 배아 수", "Zero transfer is branch information.", "Transfer count table.", "May dominate target.", "Binary flag.", "high"),
        ("transfer_day_missing", "배아 이식 경과일", "Missing day is informative.", "Missingness branch table.", "Structural missingness.", "Missing flag plus bins.", "high"),
        ("transfer_day_5", "배아 이식 경과일", "Day 5 proxies blastocyst transfer.", "Transfer day table.", "Embryo quality confounding.", "Binary flag.", "medium"),
        ("possible_blastocyst_transfer", "배아 이식 경과일; 특정 시술 유형", "Blastocyst proxy combines explicit and timing cues.", "Day and token tables.", "Proxy only.", "Derived flag.", "medium"),
        ("reason_current_treatment", "배아 생성 주요 이유", "Current treatment cycles differ from storage/donation.", "Reason token table.", "Text categories may be multi-label.", "Token flag.", "high"),
        ("storage_only_flag", "배아 생성 주요 이유", "Storage-only branch may not target immediate pregnancy.", "Reason branch table.", "Branch definition is heuristic.", "Flag and review.", "high"),
        ("current_treatment_absent_flag", "배아 생성 주요 이유", "Absent current treatment reason may identify non-treatment cycles.", "Reason branch table.", "Text may be incomplete.", "Flag.", "medium"),
        ("oocyte_count_bin", "수집된 신선 난자 수", "Oocyte count has nonlinear plateau.", "Funnel bin table.", "Age and protocol confounding.", "Bin numeric count.", "medium"),
        ("embryo_created_bin", "총 생성 배아 수", "Embryo count is nonlinear lab signal.", "Funnel bin table.", "Post-treatment signal.", "Bin numeric count.", "medium"),
        ("embryo_creation_rate", "총 생성 배아 수; 혼합된 난자 수", "Conversion ratio may capture lab yield.", "Ratio summary.", "Denominator zero/missing.", "NaN when denominator <= 0.", "medium"),
        ("icsi_embryo_creation_rate", "미세주입에서 생성된 배아 수; 미세주입된 난자 수", "ICSI yield may matter.", "Ratio summary.", "Only ICSI branch.", "Branch-aware ratio.", "medium"),
        ("transfer_per_embryo_rate", "이식된 배아 수; 총 생성 배아 수", "Transfer intensity relative to embryos.", "Ratio bins.", "Frozen cycles can break same-cycle link.", "Use with branch flags.", "medium"),
        ("sperm_factor_count", "정자 관련 불임 원인", "Multiple sperm factors indicate severity.", "Cause group table.", "Recording quality varies.", "Count binary sperm causes.", "medium"),
        ("female_factor_count", "여성 관련 불임 원인", "Multiple female factors indicate complexity.", "Cause group table.", "Cause overlap.", "Count binary female causes.", "medium"),
        ("previous_pregnancy_flag", "총 임신 횟수", "Previous pregnancy indicates history.", "History table.", "May reflect age/treatment attempts.", "Binary from count category.", "medium"),
        ("previous_live_birth_flag", "총 출산 횟수", "Prior live birth may change prognosis.", "History table.", "Confounded with prior attempts.", "Binary from count category.", "medium"),
    ]
    return pd.DataFrame(
        items,
        columns=[
            "feature_candidate",
            "source_columns",
            "domain_hypothesis",
            "eda_evidence",
            "confounding_or_caution",
            "suggested_handling",
            "priority",
        ],
    )


def load_data_dictionary() -> pd.DataFrame:
    if DICT_PATH.exists():
        try:
            raw = pd.read_excel(DICT_PATH)
            raw.columns = [str(c) for c in raw.columns]
            return raw
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def make_data_dictionary_enriched(df: pd.DataFrame, missing_interp: pd.DataFrame) -> pd.DataFrame:
    original = load_data_dictionary()
    desc_map: dict[str, str] = {}
    type_map: dict[str, str] = {}
    if not original.empty:
        col_candidates = [c for c in original.columns if "컬럼" in c or "변수" in c or "column" in c.lower()]
        desc_candidates = [c for c in original.columns if "설명" in c or "description" in c.lower()]
        type_candidates = [c for c in original.columns if "타입" in c or "type" in c.lower()]
        if col_candidates:
            key = col_candidates[0]
            if desc_candidates:
                desc_map = dict(zip(original[key].astype(str), original[desc_candidates[0]].astype(str)))
            if type_candidates:
                type_map = dict(zip(original[key].astype(str), original[type_candidates[0]].astype(str)))
    special = {
        "시술 당시 나이": ("baseline_risk", "all", "Donor egg use can confound older age groups.", "Ordinal + category + unknown flag."),
        "이식된 배아 수": ("transfer_stage_signal", "IVF/FET", "No-transfer branch must be explicit.", "Count bin plus transfer flag."),
        "배아 이식 경과일": ("transfer_timing", "transfer cycle", "Missing itself is informative.", "Day bins plus day5/blastocyst proxy."),
        "해동된 배아 수": ("frozen_thaw_signal", "frozen/thaw", "Structural missing likely outside frozen branch.", "Use with frozen flag."),
        "배아 생성 주요 이유": ("branch_reason", "current/storage/donation", "Multi-label text branch signal.", "Token flags."),
    }
    rows = []
    for col in df.columns:
        interp = missing_interp[missing_interp["column"].eq(col)].iloc[0]
        role, branch, confound, handling = special.get(
            col,
            (
                "target" if col == "임신 성공 여부" else ("identifier" if col == "ID" else "candidate_feature"),
                "all",
                interp["interpretation"],
                interp["feature_handling_note"],
            ),
        )
        rows.append(
            {
                "column": col,
                "original_type": type_map.get(col, str(df[col].dtype)),
                "original_description": desc_map.get(col, ""),
                "inferred_role": role,
                "branch_dependency": branch,
                "likely_structural_missing": interp["likely_structural_missing"],
                "confounding_note": confound,
                "feature_handling_note": handling,
            }
        )
    return pd.DataFrame(rows)


def df_preview(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(n).to_html(index=False, escape=True, border=0)


def build_html_report(tables: dict[str, pd.DataFrame], figures: list[str], schema: dict[str, object]) -> None:
    css = """
    body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#222}
    h1,h2{color:#1f3b4d} table{border-collapse:collapse;font-size:12px;margin:12px 0 24px}
    th,td{border:1px solid #ddd;padding:6px 8px;text-align:left} th{background:#f2f5f7}
    img{max-width:100%;height:auto;border:1px solid #ddd;margin:8px 0 20px}
    .note{background:#fff8dc;border-left:4px solid #c99700;padding:10px 12px}
    """
    sections = [
        ("Executive Summary", "Train-only EDA generated branch-aware ART cycle summaries, structural missingness checks, funnel signals, and feature hypotheses."),
        ("Leakage-safe EDA 원칙", LEAKAGE_NOTE),
        ("Dataset Overview", df_preview(tables.get("overview", pd.DataFrame()))),
        ("Why this is cycle-level ART data", "Rows contain treatment type, gamete/embryo handling, transfer timing, storage/thaw signals, and outcome fields, so cycle-level branch interpretation is required."),
        ("Treatment Branch Structure", df_preview(tables.get("target_rate_by_treatment_type", pd.DataFrame()))),
        ("Age and Egg Source Confounding", "Age is a core ART predictor, but donor egg use can strongly confound older age groups." + df_preview(tables.get("target_rate_by_age_egg_source", pd.DataFrame()))),
        ("Structural Missingness", df_preview(tables.get("missingness_interpretation", pd.DataFrame()))),
        ("ART Funnel", df_preview(tables.get("funnel_numeric_summary", pd.DataFrame()))),
        ("Embryo Transfer Signals", df_preview(tables.get("target_rate_by_transfer_day", pd.DataFrame()))),
        ("Specific Treatment Type Token Analysis", df_preview(tables.get("target_rate_by_specific_treatment_token", pd.DataFrame()))),
        ("Embryo Creation Reason Analysis", df_preview(tables.get("target_rate_by_embryo_reason_token", pd.DataFrame()))),
        ("Treatment/Pregnancy/Birth History", df_preview(tables.get("history_target_rate", pd.DataFrame()))),
        ("Infertility Cause Aggregation", df_preview(tables.get("infertility_cause_group_target_rate", pd.DataFrame()))),
        ("Feature Engineering Implications", df_preview(tables.get("feature_hypothesis_table", pd.DataFrame()))),
        ("Next-step Checklist", "After this EDA, design leakage-safe preprocessing fitted on train folds only, then evaluate model candidates. This run intentionally did not train models, run CV, tune Optuna, or create submissions."),
    ]
    figure_html = "".join(f'<h3>{html.escape(name)}</h3><img src="figures/{html.escape(name)}" alt="{html.escape(name)}">' for name in figures)
    body = [f"<style>{css}</style>", "<h1>ART Cycle Train-only EDA Report</h1>", f'<p class="note">{LEAKAGE_NOTE}</p>']
    body.append(f"<p>Schema validation summary: {html.escape(str(schema))}</p>")
    for title, content in sections:
        body.append(f"<h2>{html.escape(title)}</h2>")
        body.append(content if content.startswith("<") else f"<p>{html.escape(content)}</p>")
    body.append("<h2>Figures</h2>")
    body.append(figure_html)
    (OUT_DIR / "eda_report.html").write_text("\n".join(body), encoding="utf-8")


def make_log(train: pd.DataFrame, test: pd.DataFrame | None, target: str, schema: dict[str, object], outputs: list[Path], warnings: list[str]) -> None:
    import matplotlib
    import pandas

    lines = [
        f"run_time: {datetime.now().isoformat(timespec='seconds')}",
        f"python_version: {platform.python_version()}",
        f"pandas_version: {pandas.__version__}",
        f"numpy_version: {np.__version__}",
        f"matplotlib_version: {matplotlib.__version__}",
        f"train_path: {TRAIN_PATH}",
        f"test_path: {TEST_PATH if TEST_PATH.exists() else 'missing'}",
        f"data_dictionary_path: {DICT_PATH if DICT_PATH.exists() else 'missing'}",
        f"train_shape: {train.shape}",
        f"test_shape_schema_only: {None if test is None else test.shape}",
        f"target_column: {target}",
        f"schema_validation: {schema}",
        f"leakage_note: {LEAKAGE_NOTE}",
        "outputs:",
    ]
    lines.extend(f"- {p.relative_to(PROJECT_ROOT)}" for p in outputs)
    lines.append("warnings:")
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- none")
    (OUT_DIR / "eda_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    train, test = load_data()
    schema = validate_schema_only(train, test)
    target = detect_target(train)
    features = train.drop(columns=[target])
    flags = make_branch_flags(features)
    eda = pd.concat([features, flags, train[[target]]], axis=1)

    save_table("overview", make_overview(train, target, schema), tables)
    missing = summarize_missingness(train)
    save_table("missingness_overall", missing, tables)
    unique = pd.DataFrame({"column": train.columns, "unique_count": train.nunique(dropna=True).values}).sort_values("unique_count", ascending=False)
    save_table("unique_counts", unique, tables)

    constant = unique[unique["unique_count"].le(1)].copy()
    save_table("constant_or_near_constant_columns", constant, tables)

    for name, col in [
        ("target_rate_by_age", "시술 당시 나이"),
        ("target_rate_by_age_egg_source", ["시술 당시 나이", "난자 출처"]),
        ("target_rate_by_age_sperm_source", ["시술 당시 나이", "정자 출처"]),
        ("target_rate_by_age_fresh_frozen", ["시술 당시 나이", "is_fresh_embryo", "is_frozen_embryo"]),
        ("target_rate_by_age_embryo_transfer_count", ["시술 당시 나이", "이식된 배아 수"]),
        ("target_rate_by_age_transfer_day", ["시술 당시 나이", "배아 이식 경과일"]),
        ("target_rate_by_treatment_type", "시술 유형"),
        ("target_rate_by_embryo_transfer_count", "이식된 배아 수"),
        ("target_rate_by_single_embryo_transfer", "단일 배아 이식 여부"),
        ("target_rate_by_transfer_day", "배아 이식 경과일"),
        ("target_rate_by_transfer_count_day", ["이식된 배아 수", "배아 이식 경과일"]),
        ("target_rate_by_fresh_frozen_transfer_day", ["is_fresh_embryo", "is_frozen_embryo", "배아 이식 경과일"]),
        ("target_rate_by_blastocyst_transfer_day", ["has_blastocyst", "배아 이식 경과일"]),
    ]:
        save_table(name, summarize_target_rate(eda, col, target), tables)

    age_unknown_mask = as_text(train.get("시술 당시 나이", pd.Series(index=train.index))).str.contains("알 수|미상|unknown", case=False, regex=True, na=False)
    age_unknown_profile = pd.DataFrame(
        {
            "metric": [
                "n",
                "target_rate",
                "top_treatment_type",
                "top_egg_source",
                "top_sperm_source",
                "transfer_day_missing_rate",
                "top_embryo_reason",
                "fresh_embryo_rate",
                "frozen_embryo_rate",
            ],
            "value": [
                int(age_unknown_mask.sum()),
                train.loc[age_unknown_mask, target].mean(),
                train.loc[age_unknown_mask, "시술 유형"].mode().iloc[0] if age_unknown_mask.any() and "시술 유형" in train else "",
                train.loc[age_unknown_mask, "난자 출처"].mode().iloc[0] if age_unknown_mask.any() and "난자 출처" in train else "",
                train.loc[age_unknown_mask, "정자 출처"].mode().iloc[0] if age_unknown_mask.any() and "정자 출처" in train else "",
                train.loc[age_unknown_mask, "배아 이식 경과일"].isna().mean() if age_unknown_mask.any() and "배아 이식 경과일" in train else np.nan,
                train.loc[age_unknown_mask, "배아 생성 주요 이유"].mode().iloc[0] if age_unknown_mask.any() and "배아 생성 주요 이유" in train else "",
                flags.loc[age_unknown_mask, "is_fresh_embryo"].mean() if age_unknown_mask.any() else np.nan,
                flags.loc[age_unknown_mask, "is_frozen_embryo"].mean() if age_unknown_mask.any() else np.nan,
            ],
        }
    )
    save_table("age_unknown_profile", age_unknown_profile, tables)

    focus_missing_cols = [
        c
        for c in [
            "임신 시도 또는 마지막 임신 경과 연수",
            "착상 전 유전 검사 사용 여부",
            "착상 전 유전 진단 사용 여부",
            "PGD 시술 여부",
            "PGS 시술 여부",
            "난자 채취 경과일",
            "난자 해동 경과일",
            "난자 혼합 경과일",
            "배아 이식 경과일",
            "배아 해동 경과일",
            "해동된 배아 수",
            "해동 난자 수",
            "이식된 배아 수",
        ]
        if c in train.columns
    ]
    miss_branch = branch_missingness(train, flags, focus_missing_cols)
    save_table("missingness_by_branch", miss_branch, tables)
    miss_interp = make_missingness_interpretation(train, missing)
    save_table("missingness_interpretation", miss_interp, tables)

    funnel_cols = [
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
    funnel_cols = [c for c in funnel_cols if c in train.columns]
    save_table("funnel_numeric_summary", numeric_summary(train, funnel_cols, target), tables)

    bin_rows = []
    for col in ["수집된 신선 난자 수", "혼합된 난자 수", "미세주입된 난자 수", "총 생성 배아 수", "이식된 배아 수", "저장된 배아 수", "해동된 배아 수"]:
        if col in train.columns:
            temp = pd.DataFrame({"column": col, "bin": bin_numeric(train[col]), target: train[target]})
            bin_rows.append(summarize_target_rate(temp, ["column", "bin"], target))
    save_table("funnel_target_rate_bins", pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame(), tables)

    ratios = make_funnel_features_for_eda(train)
    ratio_df = pd.concat([ratios, train[[target]]], axis=1)
    save_table("funnel_ratio_summary", numeric_summary(ratio_df, ratios.columns.tolist(), target), tables)
    ratio_bin_rows = []
    for col in ratios.columns:
        temp = pd.DataFrame({"column": col, "bin": pd.qcut(ratios[col], q=5, duplicates="drop").astype(str), target: train[target]})
        temp.loc[ratios[col].isna(), "bin"] = "missing"
        ratio_bin_rows.append(summarize_target_rate(temp, ["column", "bin"], target))
    save_table("funnel_ratio_target_rate_bins", pd.concat(ratio_bin_rows, ignore_index=True) if ratio_bin_rows else pd.DataFrame(), tables)
    inconsistency = pd.DataFrame(
        {
            "condition": ["이식된 배아 수 > 총 생성 배아 수"],
            "n": [
                int(
                    (
                        pd.to_numeric(train.get("이식된 배아 수", pd.Series(index=train.index)), errors="coerce")
                        > pd.to_numeric(train.get("총 생성 배아 수", pd.Series(index=train.index)), errors="coerce")
                    ).sum()
                )
            ],
            "target_rate": [
                train.loc[
                    pd.to_numeric(train.get("이식된 배아 수", pd.Series(index=train.index)), errors="coerce")
                    > pd.to_numeric(train.get("총 생성 배아 수", pd.Series(index=train.index)), errors="coerce"),
                    target,
                ].mean()
            ],
        }
    )
    save_table("funnel_inconsistency_profile", inconsistency, tables)

    specific_token = tokenize_specific(train.get("특정 시술 유형", pd.Series(index=train.index, dtype=object)))
    temp = pd.DataFrame({"specific_treatment_token": specific_token, "token_count": flags["specific_treatment_token_count"], "raw": train.get("특정 시술 유형"), target: train[target]})
    save_table("target_rate_by_specific_treatment_raw", summarize_target_rate(temp, "raw", target), tables)
    save_table("target_rate_by_specific_treatment_token", summarize_target_rate(temp, "specific_treatment_token", target), tables)
    save_table("target_rate_by_specific_treatment_token_count", summarize_target_rate(temp, "token_count", target), tables)
    save_table("target_rate_by_specific_treatment_combination", summarize_target_rate(temp, "specific_treatment_token", target), tables)

    reason_token = tokenize_reason(train.get("배아 생성 주요 이유", pd.Series(index=train.index, dtype=object)))
    temp = pd.DataFrame(
        {
            "embryo_reason_token": reason_token,
            "reason_token_count": flags["reason_token_count"],
            "raw": train.get("배아 생성 주요 이유"),
            "storage_only_flag": flags["storage_only_flag"],
            "donation_only_flag": flags["reason_donation"] & ~flags["reason_current_treatment"],
            "reason_current_treatment": flags["reason_current_treatment"],
            target: train[target],
        }
    )
    save_table("target_rate_by_embryo_reason_raw", summarize_target_rate(temp, "raw", target), tables)
    save_table("target_rate_by_embryo_reason_token", summarize_target_rate(temp, "embryo_reason_token", target), tables)
    save_table("target_rate_by_embryo_reason_token_count", summarize_target_rate(temp, "reason_token_count", target), tables)
    save_table("target_rate_by_embryo_reason_branch", summarize_target_rate(temp, ["storage_only_flag", "donation_only_flag", "reason_current_treatment"], target), tables)

    history_cols = [c for c in ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"] if c in train.columns]
    hist_rows = []
    for col in history_cols:
        temp = summarize_target_rate(train, col, target)
        temp.insert(0, "history_column", col)
        hist_rows.append(temp.rename(columns={col: "value"}))
    save_table("history_target_rate", pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame(), tables)
    hist_feat = pd.DataFrame(
        {
            "feature": ["previous_pregnancy_flag", "previous_live_birth_flag", "repeated_treatment_flag", "repeated_failure_proxy"],
            "source": ["총 임신 횟수", "총 출산 횟수", "총 시술 횟수", "총 시술 횟수; 총 임신 횟수"],
            "note": [
                "Any previous pregnancy count greater than none/zero.",
                "Any previous live birth count greater than none/zero.",
                "Multiple previous treatment attempts.",
                "Many attempts with no previous pregnancy may proxy repeated failure.",
            ],
        }
    )
    save_table("history_feature_candidates", hist_feat, tables)

    cause_cols = [c for c in train.columns if "불임 원인" in c]
    cause_rows = []
    for col in cause_cols:
        temp = summarize_target_rate(train, col, target)
        temp.insert(0, "cause_column", col)
        cause_rows.append(temp.rename(columns={col: "value"}))
    save_table("infertility_cause_target_rate", pd.concat(cause_rows, ignore_index=True) if cause_rows else pd.DataFrame(), tables)
    cause_flags = pd.DataFrame(index=train.index)
    male_cols = [c for c in cause_cols if "남성" in c or "정자" in c]
    female_cols = [c for c in cause_cols if "여성" in c or "난관" in c or "배란" in c or "자궁" in c]
    couple_cols = [c for c in cause_cols if "부부" in c]
    unexplained_cols = [c for c in cause_cols if "불명확" in c]
    cause_flags["sperm_factor_count"] = train[male_cols].apply(lambda r: pd.to_numeric(r, errors="coerce").fillna(0).gt(0).sum(), axis=1) if male_cols else 0
    cause_flags["female_factor_count"] = train[female_cols].apply(lambda r: pd.to_numeric(r, errors="coerce").fillna(0).gt(0).sum(), axis=1) if female_cols else 0
    cause_flags["couple_factor_any"] = train[couple_cols].apply(lambda r: pd.to_numeric(r, errors="coerce").fillna(0).gt(0).any(), axis=1) if couple_cols else False
    cause_flags["unexplained_factor"] = train[unexplained_cols].apply(lambda r: pd.to_numeric(r, errors="coerce").fillna(0).gt(0).any(), axis=1) if unexplained_cols else False
    cause_flags["male_factor_any"] = cause_flags["sperm_factor_count"].gt(0)
    cause_flags["female_factor_any"] = cause_flags["female_factor_count"].gt(0)
    cause_flags["total_cause_count"] = cause_flags["sperm_factor_count"] + cause_flags["female_factor_count"] + cause_flags["couple_factor_any"].astype(int) + cause_flags["unexplained_factor"].astype(int)
    cause_flags["both_male_female_factor"] = cause_flags["male_factor_any"] & cause_flags["female_factor_any"]
    group_rows = []
    for col in cause_flags.columns:
        temp = pd.DataFrame({"group_feature": col, "value": cause_flags[col], target: train[target]})
        group_rows.append(summarize_target_rate(temp, ["group_feature", "value"], target))
    save_table("infertility_cause_group_target_rate", pd.concat(group_rows, ignore_index=True), tables)

    feature_hyp = make_feature_hypothesis_table()
    save_table("feature_hypothesis_table", feature_hyp, tables)
    feature_hyp.to_csv(OUT_DIR / "feature_hypothesis_table.csv", index=False, encoding="utf-8-sig")
    data_dict = make_data_dictionary_enriched(train, miss_interp)
    save_table("data_dictionary_enriched", data_dict, tables)
    data_dict.to_csv(OUT_DIR / "data_dictionary_enriched.csv", index=False, encoding="utf-8-sig")

    figures = []
    target_dist = train[target].value_counts(dropna=False).rename_axis("target").reset_index(name="n")
    plot_bar(target_dist, "target", "n", "Target distribution", FIGURE_DIR / "target_distribution.png")
    figures.append("target_distribution.png")
    for table_name, file_name, x, y, title in [
        ("target_rate_by_age", "age_target_rate.png", "시술 당시 나이", "target_rate", "Age target rate"),
        ("target_rate_by_treatment_type", "treatment_branch_target_rate.png", "시술 유형", "target_rate", "Treatment branch target rate"),
        ("missingness_overall", "missingness_overall_bar.png", "column", "missing_rate", "Missingness overall"),
        ("funnel_target_rate_bins", "funnel_bins_target_rate.png", "bin", "target_rate", "Funnel bin target rate"),
        ("target_rate_by_embryo_transfer_count", "embryo_transferred_count_target_rate.png", "이식된 배아 수", "target_rate", "Embryo transferred count target rate"),
        ("target_rate_by_transfer_day", "transfer_day_target_rate.png", "배아 이식 경과일", "target_rate", "Transfer day target rate"),
        ("target_rate_by_age_fresh_frozen", "fresh_frozen_target_rate.png", "시술 당시 나이", "target_rate", "Fresh/frozen target rate by age"),
        ("target_rate_by_embryo_reason_token", "embryo_reason_token_target_rate.png", "embryo_reason_token", "target_rate", "Embryo reason token target rate"),
    ]:
        plot_bar(tables.get(table_name, pd.DataFrame()), x, y, title, FIGURE_DIR / file_name)
        figures.append(file_name)
    age_egg = tables.get("target_rate_by_age_egg_source", pd.DataFrame())
    if not age_egg.empty:
        pivot = age_egg.pivot_table(index="시술 당시 나이", columns="난자 출처", values="target_rate")
        plot_heatmap(pivot, "Age x egg source target rate", FIGURE_DIR / "age_egg_source_heatmap.png")
        figures.append("age_egg_source_heatmap.png")
    miss_heat = miss_branch.pivot_table(index="branch", columns="column", values="missing_rate")
    plot_heatmap(miss_heat, "Missingness by branch", FIGURE_DIR / "missingness_by_branch_heatmap.png")
    figures.append("missingness_by_branch_heatmap.png")
    spec_token = tables.get("target_rate_by_specific_treatment_token", pd.DataFrame())
    plot_bar(spec_token, "specific_treatment_token", "target_rate", "Specific treatment token target rate", FIGURE_DIR / "specific_treatment_token_target_rate.png")
    figures.append("specific_treatment_token_target_rate.png")

    save_excel_report(tables)
    build_html_report(tables, [f for f in figures if (FIGURE_DIR / f).exists()], schema)

    outputs = [
        OUT_DIR / "eda_art_cycle_analysis.py",
        OUT_DIR / "eda_report.html",
        OUT_DIR / "eda_tables.xlsx",
        OUT_DIR / "feature_hypothesis_table.csv",
        OUT_DIR / "data_dictionary_enriched.csv",
        OUT_DIR / "eda_log.txt",
        FIGURE_DIR,
        TABLE_DIR,
    ]
    make_log(train, test, target, schema, outputs, warnings)


if __name__ == "__main__":
    main()
