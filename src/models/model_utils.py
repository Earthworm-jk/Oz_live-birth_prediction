from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET_CANDIDATES = ["임신 성공 여부", "pregnancy", "target"]
ID_COLUMN = "ID"
MISSING_CATEGORY = "__MISSING__"


def get_target_column(df: pd.DataFrame) -> str:
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


def split_features_target(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    target = target_col or get_target_column(df)
    return df.drop(columns=[target]), df[target].astype(int)


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    return [
        col
        for col, dtype in X.dtypes.items()
        if dtype == "object" or str(dtype) in {"category", "bool", "boolean", "string", "str"}
    ]


def prepare_catboost_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = X.copy()
    cat_cols = get_categorical_columns(out)
    for col in cat_cols:
        out[col] = out[col].fillna(MISSING_CATEGORY).astype(str)
    return out, cat_cols


def prepare_catboost_pool(X: pd.DataFrame, y: pd.Series | None = None):
    from catboost import Pool

    prepared, cat_cols = prepare_catboost_frame(X)
    cat_indices = [prepared.columns.get_loc(c) for c in cat_cols]
    return Pool(prepared, label=y, cat_features=cat_indices), cat_cols


def compute_auc(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr[~pd.isna(y_arr)])) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, pred))


def compute_subgroup_auc(
    frame: pd.DataFrame,
    y_col: str,
    pred_col: str,
    subgroup_name: str,
    mask: pd.Series,
    subgroup_value: str = "1",
) -> dict[str, Any]:
    mask = mask.fillna(False)
    sub = frame.loc[mask]
    note = ""
    auc = float("nan")
    if len(sub) == 0:
        note = "empty subgroup"
    elif sub[y_col].nunique(dropna=True) < 2:
        note = "only one class present"
    else:
        auc = compute_auc(sub[y_col], sub[pred_col])
    return {
        "subgroup_name": subgroup_name,
        "subgroup_value": subgroup_value,
        "n": int(len(sub)),
        "positive_rate": float(sub[y_col].mean()) if len(sub) else float("nan"),
        "auc": auc,
        "note": note,
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
