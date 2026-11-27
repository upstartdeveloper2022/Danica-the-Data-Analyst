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


def test_kpis_exclude_identifiers_and_boolean_indicators():
    df = pd.DataFrame(
        {
            "PatientId": [100001, 100001, 100002, 100003],
            "AppointmentID": [900001, 900002, 900003, 900004],
            "Age": [21, 45, 37, 60],
            "Date.diff": [0, 2, 4, 1],
            "Scholarship": [False, True, False, True],
            "SMS_received": [0, 1, 0, 1],
        }
    )
    result = identify_and_compute_kpis(df)
    metrics = [kpi["metric"] for kpi in result["kpis"]]

    assert metrics == ["Age", "Date.diff"]
    assert all(kpi["aggregation"] == "mean" for kpi in result["kpis"])

    reimbursement = identify_and_compute_kpis(
        pd.DataFrame({"InscClaimAmtReimbursed": [100, 200]})
    )
    assert reimbursement["kpis"][0]["label"] == "Insc Claim Amt Reimbursed"

    identifier_only = identify_and_compute_kpis(
        pd.DataFrame({"AppointmentID": [900001, 900002, 900003]})
    )
    assert [kpi["metric"] for kpi in identifier_only["kpis"]] == ["AppointmentID"]


def test_kpis_exclude_standalone_calendar_components():
    df = pd.DataFrame(
        {
            "Year": [2023, 2024, 2025],
            "Month": [1, 2, 3],
            "Day": [10, 11, 12],
            "Effective Federal Funds Rate": [4.5, 4.7, 4.8],
            "Unemployment Rate": [3.8, 3.9, 4.0],
        }
    )

    metrics = [
        kpi["metric"]
        for kpi in identify_and_compute_kpis(df)["kpis"]
    ]

    assert "Year" not in metrics
    assert "Month" not in metrics
    assert "Day" not in metrics
    assert metrics[:2] == ["Effective Federal Funds Rate", "Unemployment Rate"]


def test_kpis_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        identify_and_compute_kpis(pd.DataFrame())
