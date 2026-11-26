from pathlib import Path

import pandas as pd
import pytest

from data_loader import load_dataset
from kpi_engine import identify_and_compute_kpis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_kpis_on_sample_operations_data():
    df = load_dataset(SAMPLE_CSV)
    result = identify_and_compute_kpis(df, date_col="date")

    metric_names = [item["metric"] for item in result["kpis"]]
    assert "tickets_received" in metric_names
    assert "operational_cost_usd" in metric_names
    assert "sla_met_pct" in metric_names

    sla_kpi = next(item for item in result["kpis"] if item["metric"] == "sla_met_pct")
    assert sla_kpi["aggregation"] == "mean"
    assert sla_kpi["delta"] is not None


def test_kpis_without_date_column():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )
    result = identify_and_compute_kpis(df, date_col=None)
    assert len(result["kpis"]) == 1
    assert result["kpis"][0]["primary_value"] == 600.0
    assert result["kpis"][0]["delta"] is None


def test_kpis_no_numeric_columns():
    df = pd.DataFrame({"status": ["Open", "Closed"], "owner": ["Alex", "Sam"]})
    result = identify_and_compute_kpis(df)
    assert result["kpis"] == []
    assert len(result["notes"]) == 1


def test_kpis_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        identify_and_compute_kpis(pd.DataFrame())