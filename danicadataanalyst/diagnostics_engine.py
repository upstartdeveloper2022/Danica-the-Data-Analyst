from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def sanitize_label(
    value: Any,
    max_len: int = 60,
    fallback: str = "Unknown",
) -> str:
    """Collapse whitespace and Markdown markers, then shorten a display label."""
    text = re.sub(
        r"\s+",
        " ",
        "" if value is None else str(value),
    ).replace("**", "").replace("__", "").strip()
    if not text:
        return fallback
    if len(text) <= max_len:
        return text

    if max_len <= 3:
        return text[:max_len]
    prefix = text[: max_len - 3].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not prefix:
        prefix = text[: max_len - 3].rstrip()
    return f"{prefix}..."


def is_free_text_column(series: pd.Series) -> bool:
    """Identify prose-like text using length, multiline, and uniqueness signals."""
    values = series.dropna().astype(str)
    if values.empty:
        return False

    average_length = float(values.str.len().mean())
    multiline_ratio = float(values.str.contains(r"[\r\n]", regex=True).mean())
    uniqueness_ratio = float(values.nunique(dropna=True) / len(values))

    return (
        average_length >= 60
        or (multiline_ratio >= 0.10 and average_length >= 20)
        or (uniqueness_ratio >= 0.80 and average_length >= 25)
        or (uniqueness_ratio >= 0.50 and average_length >= 40)
    )


def is_identifier_column(series: pd.Series, column_name: str) -> bool:
    """Identify named ID/hash fields and highly unique opaque string keys."""
    camel_case_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column_name)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_case_name.lower()).strip("_")
    tokens = set(normalized.split("_"))
    if (
        normalized == "id"
        or normalized.endswith("_id")
        or tokens & {"uuid", "guid", "hash", "identifier", "code", "key"}
    ):
        return True

    values = series.dropna().astype(str)
    if len(values) < 10:
        return False
    unique_count = int(values.nunique(dropna=True))
    uniqueness_ratio = float(unique_count / len(values))
    average_length = float(values.str.len().mean())
    if uniqueness_ratio >= 0.98 and 8 <= average_length <= 128:
        if pd.api.types.is_numeric_dtype(series):
            return True
        compact_value_ratio = float(
            values.str.fullmatch(r"[A-Za-z0-9_-]+").mean()
        )
        if compact_value_ratio >= 0.95:
            return True
    return (
        bool(tokens & {"group"})
        and unique_count >= 100
        and uniqueness_ratio >= 0.10
    )


def is_boolean_column(series: pd.Series) -> bool:
    """Identify native booleans and numeric 0/1 indicator fields."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False

    values = pd.to_numeric(series, errors="coerce").dropna()
    return not values.empty and values.isin([0, 1]).all()


def has_analytical_numeric_columns(df: pd.DataFrame) -> bool:
    """Return whether the frame has a continuous, non-identifier numeric measure."""
    return any(
        pd.api.types.is_numeric_dtype(df[col])
        and not is_boolean_column(df[col])
        and not is_identifier_column(df[col], str(col))
        for col in df.columns
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


def _analyze_segment_drivers(
    df: pd.DataFrame,
    metric_col: str,
    category_cols: list[str],
    is_rate: bool,
) -> list[dict[str, Any]]:
    """Rank categorical segments by share of total (for sums) or gap vs. overall mean (for rates)."""
    drivers: list[dict[str, Any]] = []
    clean_metric = pd.to_numeric(df[metric_col], errors="coerce")
    overall_total = float(clean_metric.sum())
    overall_mean = float(clean_metric.mean())

    for cat_col in category_cols:
        if cat_col not in df.columns:
            continue
        working = pd.DataFrame(
            {
                "segment": df[cat_col].fillna("Missing").astype(str),
                "value": clean_metric,
            }
        ).dropna(subset=["value"])

        if working.empty or working["segment"].nunique() < 2:
            continue

        grouped = working.groupby("segment", as_index=False)["value"].agg(["count", "sum", "mean"])
        if is_rate:
            grouped["diff_vs_overall"] = grouped["mean"] - overall_mean
            grouped = grouped.reindex(
                grouped["diff_vs_overall"].abs().sort_values(ascending=False).index
            )
            top = grouped.iloc[0]
            drivers.append(
                {
                    "dimension": sanitize_label(cat_col),
                    "top_segment": sanitize_label(top["segment"]),
                    "record_count": int(top["count"]),
                    "segment_value": round(float(top["mean"]), 2),
                    "overall_baseline": round(overall_mean, 2),
                    "impact_metric": f"{float(top['diff_vs_overall']):+.2f} vs. overall mean",
                }
            )
        else:
            grouped["share_pct"] = (
                (grouped["sum"] / overall_total * 100.0) if overall_total != 0 else 0.0
            )
            grouped = grouped.sort_values("sum", ascending=False)
            top = grouped.iloc[0]
            drivers.append(
                {
                    "dimension": sanitize_label(cat_col),
                    "top_segment": sanitize_label(top["segment"]),
                    "record_count": int(top["count"]),
                    "segment_value": round(float(top["sum"]), 2),
                    "overall_baseline": round(overall_total, 2),
                    "impact_metric": f"{float(top['share_pct']):.1f}% of total",
                }
            )

    return drivers


def _analyze_period_movers(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None,
    category_cols: list[str],
    is_rate: bool,
) -> list[dict[str, Any]]:
    """Identify which segment shifted most between the prior and latest weekly period."""
    if not date_col or date_col not in df.columns or not category_cols:
        return []

    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[date_col, metric_col]).sort_values(date_col)

    if working.empty:
        return []

    period_series = working[date_col].dt.to_period("W")
    periods = sorted(period_series.unique())
    if len(periods) < 2:
        return []

    prev_p, curr_p = periods[-2], periods[-1]
    prev_df = working.loc[period_series == prev_p]
    curr_df = working.loc[period_series == curr_p]

    agg_func = "mean" if is_rate else "sum"
    movers: list[dict[str, Any]] = []

    for cat_col in category_cols:
        if cat_col not in working.columns:
            continue
        prev_grp = prev_df.groupby(cat_col)[metric_col].agg(agg_func)
        curr_grp = curr_df.groupby(cat_col)[metric_col].agg(agg_func)
        combined = pd.DataFrame({"prev": prev_grp, "curr": curr_grp}).dropna()
        if combined.empty:
            continue

        combined["delta"] = combined["curr"] - combined["prev"]
        combined["pct_change"] = np.where(
            combined["prev"] != 0,
            (combined["delta"] / combined["prev"].abs()) * 100.0,
            np.nan,
        )
        top_idx = combined["delta"].abs().idxmax()
        row = combined.loc[top_idx]
        movers.append(
            {
                "dimension": sanitize_label(cat_col),
                "segment": sanitize_label(top_idx),
                "comparison": f"{curr_p} vs. {prev_p}",
                "previous": round(float(row["prev"]), 2),
                "current": round(float(row["curr"]), 2),
                "delta": round(float(row["delta"]), 2),
                "pct_change": round(float(row["pct_change"]), 2)
                if pd.notna(row["pct_change"])
                else None,
            }
        )

    return movers


def _detect_iqr_outliers(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None = None,
    category_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Flag rows where metric_col falls outside 1.5 * IQR."""
    if (
        is_identifier_column(df[metric_col], metric_col)
        and has_analytical_numeric_columns(df)
    ):
        return pd.DataFrame()

    series = pd.to_numeric(df[metric_col], errors="coerce")
    clean = series.dropna()
    if len(clean) < 4:
        return pd.DataFrame()

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return pd.DataFrame()

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (series < lower_bound) | (series > upper_bound)
    if not mask.any():
        return pd.DataFrame()

    keep_cols: list[str] = []
    if date_col and date_col in df.columns:
        keep_cols.append(date_col)
    if category_cols:
        keep_cols.extend([c for c in category_cols if c in df.columns and c not in keep_cols])
    keep_cols.append(metric_col)

    outliers = df.loc[mask, keep_cols].copy()
    outliers["flag_reason"] = np.where(
        series.loc[mask] > upper_bound,
        f"Above upper fence ({upper_bound:.2f})",
        f"Below lower fence ({lower_bound:.2f})",
    )
    return outliers.reset_index(drop=True)


def run_root_cause_diagnostics(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str | None = None,
    category_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Run driver attribution, period mover detection, and IQR outlier checks."""
    if df is None or df.empty:
        raise ValueError("Cannot run diagnostics on an empty DataFrame.")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")

    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Metric column '{metric_col}' contains no valid numeric values.")
    if is_identifier_column(df[metric_col], metric_col) and has_analytical_numeric_columns(df):
        raise ValueError(
            f"Identifier column '{metric_col}' cannot be used for diagnostics when analytical numeric measures are available."
        )

    if category_cols is None:
        category_cols = [
            col
            for col in df.columns
            if col != date_col
            and (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
                or isinstance(df[col].dtype, pd.CategoricalDtype)
            )
            and not is_free_text_column(df[col])
            and not is_identifier_column(df[col], str(col))
        ]
    else:
        category_cols = [
            col
            for col in category_cols
            if col in df.columns
            and not is_free_text_column(df[col])
        ]

    is_rate = _is_rate_metric(metric_col, series)
    drivers = _analyze_segment_drivers(df, metric_col, category_cols, is_rate=is_rate)
    movers = _analyze_period_movers(df, metric_col, date_col, category_cols, is_rate=is_rate)
    outliers_df = _detect_iqr_outliers(
        df, metric_col, date_col=date_col, category_cols=category_cols
    )

    findings: list[str] = []
    for drv in drivers:
        if is_rate:
            if drv["segment_value"] > drv["overall_baseline"]:
                segment_phrase = "leads with"
            elif drv["segment_value"] < drv["overall_baseline"]:
                segment_phrase = "lags with"
            else:
                segment_phrase = "matches the overall mean at"
        else:
            segment_phrase = "leads with"
        findings.append(
            f"By {drv['dimension']}, {drv['top_segment']} {segment_phrase} "
            f"{drv['segment_value']:,} ({drv['impact_metric']})."
        )
    for mov in movers:
        pct_txt = f" ({mov['pct_change']:+.1f}%)" if mov["pct_change"] is not None else ""
        findings.append(
            f"Largest recent shift in {mov['dimension']}: {mov['segment']} changed by "
            f"{mov['delta']:+.2f}{pct_txt} ({mov['comparison']})."
        )
    if not outliers_df.empty:
        findings.append(
            f"Detected {len(outliers_df)} statistical outlier row(s) for {metric_col} using the 1.5×IQR rule."
        )
    else:
        findings.append(f"No extreme statistical outliers detected for {metric_col} (1.5×IQR rule).")

    if len(df) < 25:
        findings.append(
            f"Caution: Small sample size ({len(df)} rows) limits statistical confidence in segment shifts."
        )

    return {
        "metric": metric_col,
        "is_rate": is_rate,
        "segment_drivers": drivers,
        "period_movers": movers,
        "outliers": outliers_df,
        "findings": findings,
    }
