from pathlib import Path

import pandas as pd
import pytest

from data_loader import load_dataset
from diagnostics_engine import (
    is_free_text_column,
    is_identifier_column,
    run_root_cause_diagnostics,
    sanitize_label,
)

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


def test_missing_segments_are_excluded_from_ranked_driver_callouts():
    df = pd.DataFrame(
        {
            "department": ["Missing"] * 4 + [None] * 2 + ["Clinic A"] * 2 + ["Clinic B"] * 2,
            "cost": [100] * 4 + [100] * 2 + [10] * 2 + [20] * 2,
        }
    )
    diagnostics = run_root_cause_diagnostics(
        df, metric_col="cost", category_cols=["department"]
    )

    assert diagnostics["segment_drivers"][0]["top_segment"] == "Clinic B"
    assert "Missing" not in diagnostics["findings"][0]


def test_missing_segments_are_excluded_from_period_movers():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"] * 2
            ),
            "department": ["Missing", "Clinic A"] * 4,
            "cost": [100, 10, 200, 15, 300, 12, 400, 18],
        }
    )
    diagnostics = run_root_cause_diagnostics(
        df, metric_col="cost", date_col="date", category_cols=["department"]
    )

    assert diagnostics["period_movers"]
    assert all(mover["segment"] != "Missing" for mover in diagnostics["period_movers"])


def test_diagnostics_rejects_identifier_metrics_and_excludes_identifier_segments():
    df = pd.DataFrame(
        {
            "AppointmentID": [900001, 900002, 900003, 900004, 900005, 900006],
            "Age": [20, 21, 22, 23, 24, 80],
            "Gender": ["F", "M", "F", "M", "F", "M"],
        }
    )
    diagnostics = run_root_cause_diagnostics(df, metric_col="Age")
    assert diagnostics["outliers"]["Age"].tolist() == [80]
    assert [driver["dimension"] for driver in diagnostics["segment_drivers"]] == [
        "Gender"
    ]

    with pytest.raises(ValueError, match="Identifier column"):
        run_root_cause_diagnostics(df, metric_col="AppointmentID")
    assert not is_identifier_column(pd.Series(["valid", "invalid"]), "valid")


def test_diagnostics_without_categories_or_dates():
    df = pd.DataFrame({"sla_pct": [91.0, 92.5, 90.0, 93.0, 89.5]})
    diag = run_root_cause_diagnostics(df, metric_col="sla_pct")
    assert diag["is_rate"] is True
    assert diag["segment_drivers"] == []
    assert diag["period_movers"] == []


def test_diagnostics_excludes_prose_and_automatic_identifier_segments():
    prose = [
        f"Paragraph one for record {idx}.\n\nParagraph two includes more detail {idx}."
        for idx in range(12)
    ]
    df = pd.DataFrame(
        {
            "department": ["Claims", "Support"] * 6,
            "text": prose,
            "prompt_hash": [f"hash-{idx:08d}" for idx in range(12)],
            "n_words": list(range(1, 13)),
        }
    )

    assert is_free_text_column(df["text"])
    assert not is_identifier_column(df["text"], "text")
    assert is_identifier_column(df["prompt_hash"], "prompt_hash")
    diagnostics = run_root_cause_diagnostics(df, metric_col="n_words")
    assert [item["dimension"] for item in diagnostics["segment_drivers"]] == ["Department"]
    assert all("**" not in finding for finding in diagnostics["findings"])

    explicitly_requested = run_root_cause_diagnostics(
        df, metric_col="n_words", category_cols=["prompt_hash"]
    )
    assert [item["dimension"] for item in explicitly_requested["segment_drivers"]] == [
        "Prompt Hash"
    ]


def test_sanitize_label_collapses_and_truncates():
    assert sanitize_label("  **Long\n label**  ") == "Long label"
    assert sanitize_label("a " * 60, max_len=20).endswith("...")
    assert len(sanitize_label("a " * 60, max_len=20)) <= 20


def test_rate_driver_narrative_matches_delta_direction():
    df = pd.DataFrame(
        {
            "negative_group": ["low"] * 10 + ["high"] * 90,
            "equal_group": ["A"] * 5 + ["B"] * 5 + ["A"] * 45 + ["B"] * 45,
            "score": [0.0] * 10 + [70.0] * 90,
        }
    )
    diagnostics = run_root_cause_diagnostics(
        df,
        metric_col="score",
        category_cols=["negative_group", "equal_group"],
    )
    statements = diagnostics["findings"]

    negative = next(line for line in statements if "Negative Group" in line)
    equal = next(line for line in statements if "Equal Group" in line)
    assert "lags with" in negative
    assert "leads with" not in negative

    positive_df = pd.DataFrame(
        {
            "group": ["high"] * 10 + ["low"] * 90,
            "score": [100.0] * 10 + [50.0] * 90,
        }
    )
    positive_result = run_root_cause_diagnostics(
        positive_df,
        metric_col="score",
        category_cols=["group"],
    )
    positive = positive_result["findings"][0]
    assert "leads with" in positive
    assert "matches the overall mean at" in equal


def test_diagnostics_rejects_empty_or_missing_metric():
    with pytest.raises(ValueError, match="empty"):
        run_root_cause_diagnostics(pd.DataFrame(), metric_col="tickets_received")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        run_root_cause_diagnostics(df, metric_col="nonexistent")
