from __future__ import annotations

import re
from typing import Any

import pandas as pd

from diagnostics_engine import is_boolean_column, is_identifier_column

_MEAN_DEFAULT_TOKENS = {
    # Price / valuation levels
    "open",
    "high",
    "low",
    "close",
    "price",
    "nav",
    "rate",
    "unit_price",
    # Non-additive attributes: year, tempo, percentage, scores, averages
    "year",
    "release_year",
    "birth_year",
    "bpm",
    "tempo",
    "percent",
    "pct",
    "percentage",
    "ratio",
    "score",
    "rating",
    "avg",
    "average",
    "mean",
    "age",
    "diff",
}

_LOW_PRIORITY_FOCUS_TOKENS = {
    "year",
    "release_year",
    "birth_year",
    "id",
    "zip",
    "postal",
    "code",
}

_HIGH_PRIORITY_FOCUS_TOKENS = {
    "streams",
    "stream",
    "plays",
    "views",
    "revenue",
    "sales",
    "amount",
    "total",
    "count",
    "volume",
    "time_played",
    "cost",
    "spend",
    "users",
    "patients",
    "orders",
    "tickets",
}


def _tokenize_col(col_name: str) -> tuple[str, set[str]]:
    normalized = re.sub(r"[^a-z0-9]+", "_", col_name.lower()).strip("_")
    return normalized, set(normalized.split("_"))


def _default_aggregation_for_metric(col_name: str, series: pd.Series | None = None) -> str:
    """Choose 'mean' for price/year/rate/attribute columns and 'sum' for additive quantities."""
    normalized, tokens = _tokenize_col(col_name)
    if normalized in _MEAN_DEFAULT_TOKENS or (tokens & _MEAN_DEFAULT_TOKENS):
        return "mean"
    if "year" in tokens or normalized.endswith("_year"):
        return "mean"
    if series is not None and not series.empty:
        # Guard against year-like integer series (e.g., 1900..2100 with low spread)
        s_clean = pd.to_numeric(series, errors="coerce").dropna()
        if (
            not s_clean.empty
            and pd.api.types.is_integer_dtype(s_clean)
            and s_clean.min() >= 1900
            and s_clean.max() <= 2100
            and s_clean.nunique() <= 100
        ):
            return "mean"
    return "sum"


def rank_priority_metrics(numeric_cols: list[str]) -> list[str]:
    """Order numeric columns so true volume/activity metrics appear before year/ID/rate fields."""
    def _score(col: str) -> tuple[int, int]:
        normalized, tokens = _tokenize_col(col)
        if tokens & _HIGH_PRIORITY_FOCUS_TOKENS or normalized in _HIGH_PRIORITY_FOCUS_TOKENS:
            return (0, 0)
        if tokens & _LOW_PRIORITY_FOCUS_TOKENS or normalized.endswith("_year") or col.lower().endswith("_id"):
            return (2, 0)
        return (1, 0)

    return sorted(numeric_cols, key=_score)


def _format_metric_label(col_name: str) -> str:
    return col_name.replace("_", " ").strip().title()


def _compute_period_delta(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None,
    agg: str,
) -> dict[str, Any] | None:
    """Compute period-over-period change across chronological halves when a date column exists."""
    if not date_col or date_col not in df.columns:
        return None

    dates = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    valid = pd.DataFrame({"_dt": dates, "_val": vals}).dropna()
    if len(valid) < 4:
        return None

    valid = valid.sort_values("_dt").reset_index(drop=True)
    midpoint = len(valid) // 2
    prior = valid.iloc[:midpoint]["_val"]
    recent = valid.iloc[midpoint:]["_val"]

    if prior.empty or recent.empty:
        return None

    if agg == "mean":
        prior_stat = float(prior.mean())
        recent_stat = float(recent.mean())
        comparison = "prior half avg"
    else:
        prior_stat = float(prior.sum())
        recent_stat = float(recent.sum())
        comparison = "prior half sum"

    if prior_stat == 0:
        return None

    pct_change = ((recent_stat - prior_stat) / abs(prior_stat)) * 100.0
    return {
        "prior_value": round(prior_stat, 2),
        "recent_value": round(recent_stat, 2),
        "pct_change": round(float(pct_change), 2),
        "comparison": comparison,
    }


def identify_and_compute_kpis(
    df: pd.DataFrame,
    date_col: str | None = None,
    max_kpis: int = 6,
) -> dict[str, Any]:
    """Identify numeric metrics in df and compute deterministic KPI summaries."""
    kpis: list[dict[str, Any]] = []
    notes: list[str] = []

    raw_numeric_cols = [
        str(col)
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not is_boolean_column(df[col])
    ]

    if not raw_numeric_cols:
        notes.append("No continuous numeric columns detected for KPI calculation.")
        return {"kpis": kpis, "notes": notes}

    analytical_cols = [
        col
        for col in raw_numeric_cols
        if not is_identifier_column(df[col], col)
    ]
    if analytical_cols:
        candidate_cols = analytical_cols
    else:
        candidate_cols = raw_numeric_cols
        notes.append(
            "Only identifier-like numeric columns are available; using them as fallback KPIs."
        )
    ordered_cols = rank_priority_metrics(candidate_cols)

    for col in ordered_cols[:max_kpis]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        agg = _default_aggregation_for_metric(col, series)
        mean_val = float(series.mean())
        median_val = float(series.median())
        min_val = float(series.min())
        max_val = float(series.max())
        sum_val = float(series.sum())

        primary_value = mean_val if agg == "mean" else sum_val
        delta = _compute_period_delta(df, col, date_col, agg)

        kpis.append(
            {
                "metric": col,
                "label": _format_metric_label(col),
                "aggregation": agg,
                "primary_value": round(primary_value, 2),
                "sum": round(sum_val, 2),
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "delta": delta,
            }
        )

    mean_metrics = [k["metric"] for k in kpis if k["aggregation"] == "mean"]
    if mean_metrics:
        notes.append(
            "Non-additive / level metrics ("
            + ", ".join(mean_metrics)
            + ") default to mean rather than sum so years, rates, tempos, or unit/price levels aren't artificially summed."
        )

    return {"kpis": kpis, "notes": notes}
