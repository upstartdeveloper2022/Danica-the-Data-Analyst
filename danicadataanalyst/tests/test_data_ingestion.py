from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from data_loader import DataLoadError, load_dataset
from data_profiler import profile_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_load_sample_csv_and_profile():
    df = load_dataset(SAMPLE_CSV)
    profile = profile_dataset(df)

    assert profile["row_count"] == 18
    assert profile["col_count"] == 7
    assert "date" in profile["datetime_columns"]
    assert "tickets_received" in profile["numeric_columns"]
    assert "department" in profile["categorical_columns"]
    assert len(profile["readiness_notes"]) >= 1


def test_load_excel_bytes():
    df_in = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-08"],
            "category": ["A", "B"],
            "metric": [100, 200],
        }
    )
    buf = BytesIO()
    buf.name = "test_upload.xlsx"
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_in.to_excel(writer, index=False)
    buf.seek(0)

    loaded = load_dataset(buf)
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_reject_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    with pytest.raises(DataLoadError, match="Unsupported file type"):
        load_dataset(bad_file)


def test_reject_empty_csv(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="empty"):
        load_dataset(empty_file)