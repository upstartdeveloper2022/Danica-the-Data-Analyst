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
    assert is_identifier_column(df["prompt_hash"], "prompt_hash")
    diagnostics = run_root_cause_diagnostics(df, metric_col="n_words")
    assert [item["dimension"] for item in diagnostics["segment_drivers"]] == ["department"]
    assert all("**" not in finding for finding in diagnostics["findings"])

    explicitly_requested = run_root_cause_diagnostics(
        df, metric_col="n_words", category_cols=["prompt_hash"]
    )
    assert [item["dimension"] for item in explicitly_requested["segment_drivers"]] == [
        "prompt_hash"
    ]


def test_sanitize_label_collapses_and_truncates():
    assert sanitize_label("  **Long\n label**  ") == "Long label"
    assert sanitize_label("a " * 60, max_len=20).endswith("...")
    assert len(sanitize_label("a " * 60, max_len=20)) <= 20


def test_diagnostics_rejects_empty_or_missing_metric():
    with pytest.raises(ValueError, match="empty"):
        run_root_cause_diagnostics(pd.DataFrame(), metric_col="tickets_received")

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        run_root_cause_diagnostics(df, metric_col="nonexistent")
