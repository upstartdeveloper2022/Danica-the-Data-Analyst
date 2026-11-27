from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from data_loader import DataLoadError, load_dataset
from data_profiler import profile_dataset, synthesize_calendar_date

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


def test_profile_separates_identifiers_boolean_flags_and_measures():
    df = pd.DataFrame(
        {
            "PatientId": [100001, 100001, 100002],
            "AppointmentID": [900001, 900002, 900003],
            "Age": [21, 45, 37],
            "Date.diff": [0, 2, 4],
            "Scholarship": [False, True, False],
            "SMS_received": [0, 1, 0],
            "Gender": ["F", "M", "F"],
        }
    )
    profile = profile_dataset(df)

    assert profile["numeric_columns"] == ["Age", "Date.diff"]
    assert profile["boolean_columns"] == ["Scholarship", "SMS_received"]
    assert profile["identifier_columns"] == ["PatientId", "AppointmentID"]
    assert profile["categorical_columns"] == ["Gender"]
    assert profile["columns"][0]["type"] == "identifier"


def test_profile_surfaces_high_missingness_as_readiness_caveats():
    df = pd.DataFrame(
        {
            "OperatingPhysician": [None] * 8 + ["P1", "P2"],
            "Age": range(10),
        }
    )
    profile = profile_dataset(df)

    assert any(
        "Operating Physician" in note and "80.0%" in note
        for note in profile["readiness_notes"]
    )


def test_profile_caps_skew_caveats_at_five_and_summarizes_the_rest():
    df = pd.DataFrame(
        {
            f"Value{idx}": [1] * 9 + [11 + idx * 10]
            for idx in range(8)
        }
    )

    profile = profile_dataset(df)
    skew_notes = [
        note
        for note in profile["readiness_notes"]
        if "notable gap between mean" in note
    ]
    summary_notes = [
        note
        for note in profile["readiness_notes"]
        if "additional numeric columns show notable mean-vs.-median differences" in note
    ]

    assert len(skew_notes) == 5
    assert "'Value7'" in skew_notes[0]
    assert len(summary_notes) == 1
    assert "...and 3 additional numeric columns" in summary_notes[0]


def test_synthesize_date_from_calendar_parts():
    df = pd.DataFrame(
        {
            "Year": [2024, 2024],
            "Month": [2, 3],
            "Day": [29, 1],
            "metric": [10.0, 20.0],
        }
    )
    date_col = synthesize_calendar_date(df)

    assert date_col == "Date"
    assert df[date_col].tolist() == [
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-01"),
    ]


def test_synthesize_date_defaults_missing_day_to_first_of_month():
    df = pd.DataFrame({"year": [2024], "month": [7], "metric": [1.0]})

    date_col = synthesize_calendar_date(df)

    assert date_col == "Date"
    assert df[date_col].iloc[0] == pd.Timestamp("2024-07-01")


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
