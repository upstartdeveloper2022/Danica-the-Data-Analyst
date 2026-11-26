from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", text.lower()).strip()


def _match_column(question: str, columns: list[str]) -> str | None:
    """Match a column by exact name, underscore-stripped name, or word overlap."""
    q_norm = _normalize_text(question)
    q_compact = q_norm.replace(" ", "")
    q_tokens = set(q_norm.split())

    # 1. Exact or space-normalized match
    for col in columns:
        c_norm = col.lower().replace("_", " ").strip()
        c_compact = col.lower().replace("_", "")
        if c_norm and (c_norm in q_norm or c_compact in q_compact):
            return col

    # 2. Token overlap match
    best_col: str | None = None
    best_score = 0
    for col in columns:
        c_tokens = {tok for tok in col.lower().split("_") if len(tok) > 1}
        if not c_tokens:
            continue
        overlap = len(q_tokens & c_tokens)
        if overlap > best_score:
            best_score = overlap
            best_col = col

    return best_col if best_score > 0 else None


def build_suggested_questions(
    df: pd.DataFrame,
    profile: dict[str, Any],
    kpi_result: dict[str, Any],
) -> list[str]:
    """Generate dynamic example prompts tailored to the active dataset's columns."""
    suggestions: list[str] = []
    metrics = [k["metric"] for k in kpi_result.get("kpis", [])]
    categoricals = [
        c["column"] for c in profile.get("columns", []) if c.get("type") == "categorical"
    ]

    if metrics:
        m0 = metrics[0]
        suggestions.append(f"What is the total and average of {m0}?")
        suggestions.append(f"What is the forecast and recent change for {m0}?")
        suggestions.append(f"Are there any outliers in {m0}?")
        if categoricals:
            suggestions.append(f"Which {categoricals[0]} has the highest {m0}?")

    suggestions.append("Any data quality or duplicate row issues?")
    return suggestions[:5]


def answer_deterministic_question(
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any],
    kpi_result: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    forecast_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a user question deterministically using pandas and existing engine outputs."""
    q_clean = (question or "").strip()
    if not q_clean:
        return {
            "answer": "Enter a question about your dataset or pick one of the suggested prompts below.",
            "table": None,
            "matched_intent": "empty",
        }

    q_lower = _normalize_text(q_clean)
    tokens = set(q_lower.split())

    numeric_cols = [str(c) for c in profile.get("numeric_columns", []) if c in df.columns]
    kpi_metrics = [k["metric"] for k in kpi_result.get("kpis", [])]
    candidate_metrics = kpi_metrics + [c for c in numeric_cols if c not in kpi_metrics]
    categorical_cols = [
        c["column"] for c in profile.get("columns", []) if c.get("type") == "categorical"
    ]
    all_cols = [str(c) for c in df.columns]

    matched_metric = _match_column(q_clean, candidate_metrics) or (
        kpi_metrics[0] if kpi_metrics else None
    )
    matched_cat = _match_column(q_clean, categorical_cols) or (
        categorical_cols[0] if categorical_cols else None
    )
    matched_any_col = _match_column(q_clean, all_cols)

    # Intent 1: Data quality, duplicates, missingness, row/column counts
    if tokens & {"quality", "missing", "null", " nulls", "duplicate", "duplicates", "rows", "columns", "shape", "caveat", "caveats"}:
        notes = profile.get("readiness_notes", [])
        summary = (
            f"Dataset has {profile.get('row_count', len(df)):,} rows, "
            f"{profile.get('col_count', len(df.columns))} columns, and "
            f"{profile.get('duplicate_rows', 0)} duplicate rows."
        )
        if notes:
            summary += " Readiness notes: " + " | ".join(notes[:3])
        return {"answer": summary, "table": None, "matched_intent": "data_quality"}

    # Intent 2: Outliers
    if tokens & {"outlier", "outliers", "anomaly", "anomalies", "iqr", "spike", "spikes"}:
        if diagnostics is not None and not diagnostics.get("outliers", pd.DataFrame()).empty:
            out_df = diagnostics["outliers"].head(10)
            return {
                "answer": f"Found {len(diagnostics['outliers'])} outlier row(s) using the 1.5x IQR rule (showing up to 10).",
                "table": out_df,
                "matched_intent": "outliers",
            }
        return {
            "answer": "No IQR outlier rows were flagged for the active metric.",
            "table": None,
            "matched_intent": "outliers",
        }

    # Intent 3: Forecast & short-term outlook
    if tokens & {"forecast", "forecasts", "project", "projected", "projection", "next", "outlook", "predict"}:
        if forecast_result and forecast_result.get("available"):
            metric_label = forecast_result["metric"].replace("_", " ").title()
            ans = (
                f"{metric_label} latest actual is {forecast_result['last_actual']:,}, "
                f"projected to {forecast_result['next_forecast']:,} next period "
                f"({forecast_result['method']})."
            )
            return {
                "answer": ans,
                "table": forecast_result.get("forecast_df"),
                "matched_intent": "forecast",
            }
        reason = (
            forecast_result.get("reason")
            if forecast_result
            else "Forecast is unavailable for this dataset."
        )
        return {"answer": reason, "table": None, "matched_intent": "forecast"}

    # Intent 4: Top / bottom segment breakdown or driver comparison
    if (tokens & {"top", "highest", "lowest", "bottom", "most", "least", "best", "worst", "by", "segment", "segments", "driver", "drivers", "compare"}) and matched_metric and matched_cat:
        series = pd.to_numeric(df[matched_metric], errors="coerce")
        grouped = (
            df.assign(_metric_val=series)
            .groupby(matched_cat, dropna=False)["_metric_val"]
            .agg(["sum", "mean", "count"])
            .dropna(subset=["sum"])
        )
        if not grouped.empty:
            ascending = bool(tokens & {"lowest", "bottom", "least", "worst"})
            grouped = grouped.sort_values("sum", ascending=ascending).reset_index()
            total_val = grouped["sum"].sum()
            grouped["share_pct"] = (
                (grouped["sum"] / total_val * 100).round(1) if total_val else 0.0
            )
            leader = grouped.iloc[0]
            rank_word = "lowest" if ascending else "highest"
            ans = (
                f"By {matched_cat}, {leader[matched_cat]} has the {rank_word} total {matched_metric} "
                f"at {leader['sum']:,.2f} ({leader['share_pct']:.1f}% share; avg {leader['mean']:,.2f})."
            )
            return {
                "answer": ans,
                "table": grouped.head(10),
                "matched_intent": "segment_ranking",
            }

    # Intent 5: Metric summary (total/sum, average/mean, min/max, delta)
    if matched_metric:
        for kpi in kpi_result.get("kpis", []):
            if kpi["metric"] == matched_metric:
                delta_str = ""
                if kpi.get("delta") and kpi["delta"].get("pct_change") is not None:
                    delta_str = f" ({kpi['delta']['pct_change']:+.1f}% vs. prior period)"
                ans = (
                    f"{kpi['label']} ({kpi['metric']}): primary {kpi['aggregation']} is "
                    f"{kpi['primary_value']:,}{delta_str} | mean: {kpi['mean']:,}, "
                    f"median: {kpi['median']:,}, range: {kpi['min']:,} to {kpi['max']:,}."
                )
                return {"answer": ans, "table": None, "matched_intent": "metric_summary"}

    # Fallback: helpful guidance grounded in current columns
    col_preview = ", ".join(all_cols[:8])
    return {
        "answer": (
            f"I couldn't match a specific metric rule for that question. "
            f"Available columns include: {col_preview}. "
            f"Try asking about totals, averages, top segments, forecasts, outliers, or data quality."
        ),
        "table": None,
        "matched_intent": "fallback",
    }