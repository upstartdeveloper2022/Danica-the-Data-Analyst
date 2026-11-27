from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from diagnostics_engine import (
    format_column_label,
    has_analytical_numeric_columns,
    is_identifier_column,
    valid_category_mask,
)


def _is_excluded_identifier_metric(df: pd.DataFrame, metric_col: str) -> bool:
    return (
        metric_col in df.columns
        and has_analytical_numeric_columns(df)
        and is_identifier_column(df[metric_col], metric_col)
    )


def _is_rate_metric(col_name: str, series: pd.Series) -> bool:
    lowered = col_name.strip().lower()
    if any(
        token in lowered
        for token in ("pct", "percent", "rate", "sla", "ratio", "score", "avg", "mean", "age", "diff")
    ):
        return True
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if (
        not clean.empty
        and clean.min() >= 0.0
        and clean.max() <= 100.0
        and not any(token in lowered for token in ("cost", "ticket", "count", "total", "revenue", "amount"))
    ):
        return True
    return False


def build_trend_chart(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None,
) -> go.Figure | None:
    """Build a time-series trend chart if a valid date and numeric metric exist."""
    if df is None or df.empty or not date_col or date_col not in df.columns:
        return None
    if metric_col not in df.columns:
        return None
    if _is_excluded_identifier_metric(df, metric_col):
        return None

    working = df[[date_col, metric_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[date_col, metric_col]).sort_values(date_col)

    if working.empty:
        return None

    is_rate = _is_rate_metric(metric_col, working[metric_col])
    agg_func = "mean" if is_rate else "sum"
    grouped = working.groupby(date_col, as_index=False)[metric_col].agg(agg_func)

    fig = px.line(
        grouped,
        x=date_col,
        y=metric_col,
        markers=True,
        title=f"{format_column_label(metric_col)} Over Time ({agg_func})",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
    return fig


def build_breakdown_chart(
    df: pd.DataFrame,
    metric_col: str,
    category_col: str | None,
    top_n: int = 10,
) -> go.Figure | None:
    """Build a categorical bar chart for the selected metric."""
    if df is None or df.empty or not category_col or category_col not in df.columns:
        return None
    if metric_col not in df.columns:
        return None
    if _is_excluded_identifier_metric(df, metric_col):
        return None
    if (
        has_analytical_numeric_columns(df)
        and is_identifier_column(df[category_col], category_col)
    ):
        return None

    working = df[[category_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.loc[valid_category_mask(working[category_col])]
    working[category_col] = working[category_col].astype(str)
    working = working.dropna(subset=[metric_col])

    if working.empty:
        return None

    is_rate = _is_rate_metric(metric_col, working[metric_col])
    agg_func = "mean" if is_rate else "sum"
    grouped = (
        working.groupby(category_col, as_index=False)[metric_col]
        .agg(agg_func)
        .sort_values(metric_col, ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        grouped,
        x=category_col,
        y=metric_col,
        title=f"{format_column_label(metric_col)} by {format_column_label(category_col)} ({agg_func})",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
    return fig


def build_distribution_chart(
    df: pd.DataFrame,
    metric_col: str,
) -> go.Figure | None:
    """Build a histogram/distribution chart for a numeric metric."""
    if df is None or df.empty or metric_col not in df.columns:
        return None
    if _is_excluded_identifier_metric(df, metric_col):
        return None

    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if series.empty:
        return None

    fig = px.histogram(
        x=series,
        nbins=min(20, max(5, len(series))),
        title=f"Distribution of {format_column_label(metric_col)}",
        labels={"x": format_column_label(metric_col)},
    )
    min_value = float(series.min())
    max_value = float(series.max())
    if min_value == max_value:
        padding = abs(min_value) * 0.05 or 0.5
        axis_range = [min_value - padding, max_value + padding]
    else:
        axis_range = [min_value, max_value]
    fig.update_xaxes(range=axis_range)
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
    return fig


def generate_milestone3_charts(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None = None,
    category_col: str | None = None,
) -> dict[str, Any]:
    """Generate available charts and contextual notes for the active selection."""
    if df is None or df.empty:
        raise ValueError("Cannot generate charts from an empty DataFrame.")

    trend_fig = build_trend_chart(df, metric_col=metric_col, date_col=date_col)
    breakdown_fig = build_breakdown_chart(df, metric_col=metric_col, category_col=category_col)
    dist_fig = build_distribution_chart(df, metric_col=metric_col)

    notes: list[str] = []
    if trend_fig is None:
        notes.append("No trend chart rendered because no valid date column or plottable metric values were found.")
    if breakdown_fig is None:
        notes.append("No category breakdown chart rendered because no categorical column was available.")
    if len(df) < 25:
        notes.append(f"Small dataset ({len(df)} rows): visual trends and distributions should be read cautiously.")

    return {
        "trend_fig": trend_fig,
        "breakdown_fig": breakdown_fig,
        "distribution_fig": dist_fig,
        "notes": notes,
    }
