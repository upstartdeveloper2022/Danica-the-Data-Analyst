from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from data_loader import load_dataset
from forecast_engine import generate_metric_forecast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_forecast_on_sample_operations_data():
    df = load_dataset(SAMPLE_CSV)
    result = generate_metric_forecast(df, metric_col="tickets_received", date_col="date", periods_ahead=3)

    assert result["available"] is True
    assert len(result["forecast_df"]) == 3
    assert isinstance(result["chart"], go.Figure)
    assert (result["forecast_df"]["forecast_value"] >= 0).all()


def test_forecast_without_date_column():
    df = pd.DataFrame({"tickets": [100, 120, 140]})
    result = generate_metric_forecast(df, metric_col="tickets", date_col=None)
    assert result["available"] is False
    assert result["chart"] is None


def test_forecast_handles_flat_series():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="W"),
            "sla_pct": [90.0, 90.0, 90.0, 90.0, 90.0],
        }
    )
    result = generate_metric_forecast(df, metric_col="sla_pct", date_col="date", periods_ahead=2)
    assert result["available"] is True
    assert list(result["forecast_df"]["forecast_value"]) == [90.0, 90.0]


def test_forecast_skips_identifier_when_measures_exist():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="W"),
            "AppointmentID": [900001, 900002, 900003, 900004, 900005],
            "Age": [21, 45, 37, 60, 33],
        }
    )
    result = generate_metric_forecast(df, "AppointmentID", date_col="date")

    assert result["available"] is False
    assert result["forecast_df"].empty
    assert "identifier" in result["reason"]


def test_age_and_date_difference_forecasts_use_mean_aggregation():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="W"),
            "Age": [20, 45, 80, 110, 75],
            "Date.diff": [0, 5, 120, 180, 90],
        }
    )
    age = generate_metric_forecast(df, "Age", date_col="date")
    diff = generate_metric_forecast(df, "Date.diff", date_col="date")

    assert age["agg_func"] == "mean"
    assert diff["agg_func"] == "mean"


def test_forecast_rejects_empty_or_invalid_args():
    with pytest.raises(ValueError, match="empty"):
        generate_metric_forecast(pd.DataFrame(), metric_col="tickets")

    df = pd.DataFrame({"tickets": [1, 2]})
    with pytest.raises(ValueError, match="not found"):
        generate_metric_forecast(df, metric_col="missing_col")
