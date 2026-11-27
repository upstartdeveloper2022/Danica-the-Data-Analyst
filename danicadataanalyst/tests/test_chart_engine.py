from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from chart_engine import (
    build_breakdown_chart,
    build_distribution_chart,
    build_trend_chart,
    generate_milestone3_charts,
)
from data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_charts_on_sample_operations_data():
    df = load_dataset(SAMPLE_CSV)
    bundle = generate_milestone3_charts(
        df,
        metric_col="tickets_received",
        date_col="date",
        category_col="department",
    )

    assert isinstance(bundle["trend_fig"], go.Figure)
    assert isinstance(bundle["breakdown_fig"], go.Figure)
    assert isinstance(bundle["distribution_fig"], go.Figure)
    assert any("Small dataset" in n for n in bundle["notes"])


def test_charts_without_date_or_category():
    df = pd.DataFrame({"score": [10, 20, 30, 40]})
    assert build_trend_chart(df, metric_col="score", date_col=None) is None
    assert build_breakdown_chart(df, metric_col="score", category_col=None) is None
    assert isinstance(build_distribution_chart(df, metric_col="score"), go.Figure)


def test_breakdown_chart_excludes_null_and_missing_category_buckets():
    df = pd.DataFrame(
        {
            "department": ["Missing", None, "Clinic A", "Clinic B"],
            "cost": [100, 200, 10, 20],
        }
    )
    chart = build_breakdown_chart(df, metric_col="cost", category_col="department")

    assert chart is not None
    assert list(chart.data[0].x) == ["Clinic B", "Clinic A"]


def test_identifier_metrics_are_not_charted_when_measures_exist():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "AppointmentID": [900001, 900002, 900003, 900004],
            "Age": [21, 45, 37, 60],
            "group": ["A", "B", "A", "B"],
        }
    )
    assert build_trend_chart(df, metric_col="AppointmentID", date_col="date") is None
    assert build_breakdown_chart(df, metric_col="AppointmentID", category_col="group") is None
    assert build_distribution_chart(df, metric_col="AppointmentID") is None


def test_age_and_date_difference_trends_use_means():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "Age": [20, 45, 80, 110],
            "Date.diff": [0, 5, 120, 180],
        }
    )
    age_chart = build_trend_chart(df, metric_col="Age", date_col="date")
    diff_chart = build_trend_chart(df, metric_col="Date.diff", date_col="date")

    assert "Over Time (mean)" in age_chart.layout.title.text
    assert "Over Time (mean)" in diff_chart.layout.title.text


def test_distribution_axis_uses_actual_data_bounds():
    df = pd.DataFrame({"Day": [1, 2, 15, 31]})

    chart = build_distribution_chart(df, metric_col="Day")

    assert chart.layout.xaxis.range == (1.0, 31.0)


def test_generate_charts_rejects_empty_df():
    with pytest.raises(ValueError, match="empty"):
        generate_milestone3_charts(pd.DataFrame(), metric_col="tickets_received")
