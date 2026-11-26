from __future__ import annotations

import re
from typing import Any

import pandas as pd

_DURATION_STR_PATTERN = re.compile(r"^\s*\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*$")
_DURATION_NAME_HINTS = {
    "duration",
    "length",
    "elapsed",
    "time_played",
    "ms_played",
    "min",
    "mins",
    "minute",
    "minutes",
    "sec",
    "secs",
    "second",
    "seconds",
}


def _looks_like_duration_column(col_name: str, non_null: pd.Series) -> bool:
    """Detect elapsed duration strings like '3:56' (M:SS) or '2:05:55' (H:MM:SS)."""
    sample = non_null.head(100)
    if sample.empty:
        return False

    matches = sample.apply(lambda v: bool(_DURATION_STR_PATTERN.match(str(v))))
    match_ratio = float(matches.sum()) / len(sample)
    if match_ratio < 0.70:
        return False

    normalized = col_name.lower().replace("-", "_").replace(" ", "_")
    tokens = set(normalized.split("_"))
    if tokens & _DURATION_NAME_HINTS or normalized.endswith("_h") or normalized.endswith("_min"):
        return True

    # Even without name hints, colon-separated values with small first components
    # (e.g., 0-23 hours or 0-99 minutes) are durations unless they include a date/AM-PM.
    has_date_or_ampm = sample.str.contains(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|am|pm|[-/])",
        case=False,
        regex=True,
    ).any()
    return not has_date_or_ampm


def _looks_like_date_series(col_name: str, series: pd.Series) -> tuple[bool, pd.Series | None]:
    """Return (True, parsed_series) if a non-numeric series is predominantly valid dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        return False, None

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return False, None

    if _looks_like_duration_column(col_name, non_null):
        return False, None

    sample = non_null.head(200)
    parsed_sample = pd.to_datetime(sample, errors="coerce")
    if (parsed_sample.notna().sum() / len(sample)) >= 0.80:
        parsed_full = pd.to_datetime(series, errors="coerce")
        if (parsed_full.notna().sum() / len(non_null)) >= 0.80:
            return True, parsed_full

    return False, None


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    row_count, col_count = df.shape
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns: list[str] = []
    datetime_columns: list[str] = []
    columns_info: list[dict[str, Any]] = []
    numeric_summary: dict[str, dict[str, float]] = {}
    readiness_notes: list[str] = []

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        missing_count = int(series.isna().sum())

        is_dt, parsed_dt = _looks_like_date_series(col_name, series)
        if is_dt and parsed_dt is not None:
            datetime_columns.append(col_name)
            columns_info.append(
                {
                    "column": col_name,
                    "type": "datetime",
                    "dtype": str(parsed_dt.dtype),
                    "missing_count": missing_count,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col_name)
            num_s = pd.to_numeric(series, errors="coerce").dropna()
            col_type = "numeric"
            if not num_s.empty:
                mean_val = float(num_s.mean())
                median_val = float(num_s.median())
                numeric_summary[col_name] = {
                    "min": round(float(num_s.min()), 4),
                    "median": round(median_val, 4),
                    "mean": round(mean_val, 4),
                    "max": round(float(num_s.max()), 4),
                }
                if median_val != 0 and abs(mean_val - median_val) / abs(median_val) > 0.25:
                    readiness_notes.append(
                        f"'{col_name}' shows a notable gap between mean ({mean_val:.2f}) and median ({median_val:.2f}) - check for outliers or skew."
                    )
        else:
            col_type = "categorical"

        columns_info.append(
            {
                "column": col_name,
                "type": col_type,
                "dtype": str(series.dtype),
                "missing_count": missing_count,
            }
        )

    if not datetime_columns:
        readiness_notes.append(
            "No date/time column detected: time-series trends and forecasting will not be available unless a date column is added."
        )

    return {
        "row_count": int(row_count),
        "col_count": int(col_count),
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "datetime_columns": datetime_columns,
        "columns": columns_info,
        "numeric_summary": numeric_summary,
        "readiness_notes": readiness_notes,
    }