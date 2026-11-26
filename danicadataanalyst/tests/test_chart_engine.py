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


def test_generate_charts_rejects_empty_df():
    with pytest.raises(ValueError, match="empty"):
        generate_milestone3_charts(pd.DataFrame(), metric_col="tickets_received")