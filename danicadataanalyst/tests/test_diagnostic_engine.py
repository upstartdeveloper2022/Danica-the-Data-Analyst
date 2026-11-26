from pathlib import Path

import pandas as pd
import pytest

from data_loader import load_dataset
from diagnostics_engine import run_root_cause_diagnostics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_diagnostics_on_sample_operations_data():
    df = load_dataset(SAMPLE_CSV)
    diag = run_root_cause_diagnostics(
        df,
        metric_col="tickets_received",
        date_col="date",
        category_cols=["department", "region"],
    )

    assert diag["metric"] == "tickets_received"
    assert diag["is_rate"] is False
    assert len(diag["segment_drivers"]) == 2
    assert len(diag["period_movers"]) == 2
    assert len(diag["findings"]) >= 3


def test_diagnostics_detects_outliers():
    df = pd.DataFrame(
        {
            "team": ["A", "A", "B", "B", "B", "A"],
            "cost": [100, 105, 102, 98, 101, 950],
        }
    )
    diag = run_root_cause_diagnostics(df, metric_col="cost", category_cols=["team"])
    assert len(diag["outliers"]) == 1
    assert diag["outliers"].iloc[0]["cost"] == 950


def test_diagnostics_without_categories_or_dates():
    df = pd.DataFrame({"sla_pct": [91.0, 92.5, 90.0, 93.0, 89.5]})
    diag = run_root_cause_diagnostics(df, metric_col="sla_pct")
    assert diag["is_rate"] is True
    assert diag["segment_drivers"] == []
    assert diag["period_movers"] == []


def test_diagnostics_rejects_empty_or_missing_metric():
    with pytest.raises(ValueError, match="empty"):
        run_root_cause_diagnostics(pd.DataFrame(), metric_col="tickets_received")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        run_root_cause_diagnostics(df, metric_col="nonexistent")