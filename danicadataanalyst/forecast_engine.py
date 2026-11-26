from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import Holt


def _is_rate_metric(col_name: str, series: pd.Series) -> bool:
    lowered = col_name.strip().lower()
    if any(token in lowered for token in ("pct", "percent", "rate", "sla", "ratio", "score", "avg", "mean")):
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


def _infer_step_offset(dates: pd.Series) -> pd.DateOffset:
    """Infer a sensible forward date step from historical timestamps."""
    sorted_dates = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(sorted_dates) >= 2:
        diffs = sorted_dates.diff().dropna()
        median_days = float(diffs.dt.total_seconds().median() / 86400.0)
        if median_days >= 1.0:
            return pd.DateOffset(days=int(round(median_days)))
    return pd.DateOffset(weeks=1)


def build_forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    metric_col: str,
    date_col: str,
    agg_func: str,
) -> go.Figure:
    """Build a Plotly chart combining historical actuals and forecasted periods."""
    metric_label = metric_col.replace("_", " ").title()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df[date_col],
            y=history_df["value"],
            mode="lines+markers",
            name=f"Actual ({agg_func})",
        )
    )

    if not forecast_df.empty:
        # Bridge from last actual point so the forecast line connects visually
        last_actual = history_df.iloc[[-1]]
        bridge_df = pd.concat(
            [
                pd.DataFrame({date_col: last_actual[date_col], "forecast_value": last_actual["value"]}),
                forecast_df[[date_col, "forecast_value"]],
            ],
            ignore_index=True,
        )
        fig.add_trace(
            go.Scatter(
                x=bridge_df[date_col],
                y=bridge_df["forecast_value"],
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{metric_label} — Actuals & {len(forecast_df)}-Period Forecast ({agg_func})",
        xaxis_title=date_col,
        yaxis_title=f"{metric_label} ({agg_func})",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def generate_metric_forecast(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None = None,
    periods_ahead: int = 3,
) -> dict[str, Any]:
    """Generate a short-term time-series forecast for a numeric metric."""
    if df is None or df.empty:
        raise ValueError("Cannot generate a forecast from an empty DataFrame.")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")
    if periods_ahead < 1:
        raise ValueError("periods_ahead must be at least 1.")

    if not date_col or date_col not in df.columns:
        return {
            "available": False,
            "reason": "No valid date/time column available for time-series forecasting.",
            "history_df": pd.DataFrame(),
            "forecast_df": pd.DataFrame(),
            "chart": None,
            "notes": ["Forecast skipped because no date column was detected."],
        }

    working = df[[date_col, metric_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[date_col, metric_col]).sort_values(date_col)

    if working.empty or working[date_col].nunique() < 2:
        return {
            "available": False,
            "reason": "At least 2 distinct time periods with valid numeric values are required to forecast.",
            "history_df": pd.DataFrame(),
            "forecast_df": pd.DataFrame(),
            "chart": None,
            "notes": ["Forecast skipped due to insufficient distinct time periods."],
        }

    is_rate = _is_rate_metric(metric_col, working[metric_col])
    agg_func = "mean" if is_rate else "sum"

    history_df = (
        working.groupby(date_col, as_index=False)[metric_col]
        .agg(agg_func)
        .rename(columns={metric_col: "value"})
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    values = history_df["value"].astype(float).to_numpy()
    n_points = len(values)
    method_used = "Linear Trend Fallback"

    if n_points >= 4 and np.ptp(values) > 0:
        try:
            model = Holt(values, initialization_method="estimated")
            fitted = model.fit(optimized=True)
            raw_preds = np.asarray(fitted.forecast(periods_ahead), dtype=float)
            method_used = "Holt Linear Exponential Smoothing"
        except Exception:
            x_idx = np.arange(n_points, dtype=float)
            slope, intercept = np.polyfit(x_idx, values, 1)
            future_idx = np.arange(n_points, n_points + periods_ahead, dtype=float)
            raw_preds = intercept + slope * future_idx
    else:
        x_idx = np.arange(n_points, dtype=float)
        if np.ptp(values) > 0 and n_points >= 2:
            slope, intercept = np.polyfit(x_idx, values, 1)
            future_idx = np.arange(n_points, n_points + periods_ahead, dtype=float)
            raw_preds = intercept + slope * future_idx
        else:
            raw_preds = np.repeat(float(values[-1]), periods_ahead)
        method_used = "Linear Trend / Last-Value Baseline"

    # Prevent negative volumes/costs/rates where non-negative domain applies
    if float(history_df["value"].min()) >= 0.0:
        raw_preds = np.maximum(raw_preds, 0.0)

    step = _infer_step_offset(history_df[date_col])
    last_date = pd.Timestamp(history_df[date_col].iloc[-1])
    future_dates = [last_date + step * i for i in range(1, periods_ahead + 1)]

    forecast_df = pd.DataFrame(
        {
            date_col: future_dates,
            "forecast_value": np.round(raw_preds, 2),
        }
    )

    notes: list[str] = [
        f"Forecast model: **{method_used}** ({agg_func} aggregation over `{date_col}`)."
    ]
    if n_points < 8 or len(df) < 25:
        notes.append(
            f"Caution: Small history ({n_points} time points across {len(df)} rows); treat projections as directional only."
        )

    chart = build_forecast_chart(
        history_df=history_df,
        forecast_df=forecast_df,
        metric_col=metric_col,
        date_col=date_col,
        agg_func=agg_func,
    )

    return {
        "available": True,
        "metric": metric_col,
        "agg_func": agg_func,
        "method": method_used,
        "last_actual": round(float(history_df["value"].iloc[-1]), 2),
        "next_forecast": round(float(forecast_df["forecast_value"].iloc[0]), 2),
        "history_df": history_df,
        "forecast_df": forecast_df,
        "chart": chart,
        "notes": notes,
    }