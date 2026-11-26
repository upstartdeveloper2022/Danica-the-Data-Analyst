from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""


_MISSING_TOKENS = {"", "-", "--", "—", "na", "n/a", "nan", "none", "null"}


def _decode_text_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _detect_csv_delimiter(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ","

    sample = "\n".join(lines[:15])
    delimiters = [",", ";", "\t", "|"]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
        if dialect.delimiter in delimiters:
            return dialect.delimiter
    except csv.Error:
        pass

    comma_count = sample.count(",")
    best_delim = ","
    best_count = comma_count

    for delim in (";", "\t", "|"):
        count = sample.count(delim)
        if count > best_count:
            best_count = count
            best_delim = delim

    return best_delim


def _read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    text = _decode_text_bytes(raw_bytes)
    sep = _detect_csv_delimiter(text)
    df = pd.read_csv(StringIO(text), sep=sep)

    if df.shape[1] == 1:
        col_name = str(df.columns[0])
        for fallback_sep in (";", ",", "\t", "|"):
            if fallback_sep != sep and fallback_sep in col_name:
                retry_df = pd.read_csv(StringIO(text), sep=fallback_sep)
                if retry_df.shape[1] > 1:
                    return retry_df

    return df


def _coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns containing comma-formatted numbers and dash placeholders."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        if out[col].dtype != object and str(out[col].dtype) != "string":
            continue

        raw_str = out[col].astype(str).str.strip()
        non_empty = raw_str[~raw_str.str.lower().isin(_MISSING_TOKENS)]
        if non_empty.empty:
            continue

        cleaned = (
            raw_str.str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
        )
        cleaned = cleaned.where(~raw_str.str.lower().isin(_MISSING_TOKENS), other=pd.NA)
        parsed = pd.to_numeric(cleaned, errors="coerce")

        valid_after_clean = parsed.notna().sum()
        # Convert if at least 70% of non-placeholder rows parse cleanly as numbers
        if len(non_empty) > 0 and (valid_after_clean / len(non_empty)) >= 0.70:
            out[col] = parsed

    return out


def load_dataset(source: str | Path | Any) -> pd.DataFrame:
    """Load a CSV or Excel dataset from a path or uploaded file-like object."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DataLoadError(f"Expected a file, got: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = _read_csv_bytes(path.read_bytes())
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                raise DataLoadError(
                    f"Unsupported file extension '{suffix}'. Please upload a .csv, .xlsx, or .xls file."
                )
            return _coerce_numeric_like_columns(df)
        except DataLoadError:
            raise
        except Exception as exc:
            raise DataLoadError(f"Failed to read '{path.name}': {exc}") from exc

    file_name = getattr(source, "name", "uploaded_file")
    suffix = Path(file_name).suffix.lower()

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        raw_bytes = source.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
        if not isinstance(raw_bytes, (bytes, bytearray)) or len(raw_bytes) == 0:
            raise DataLoadError(f"Uploaded file '{file_name}' is empty.")

        if suffix == ".csv" or not suffix:
            df = _read_csv_bytes(bytes(raw_bytes))
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(BytesIO(bytes(raw_bytes)))
        else:
            raise DataLoadError(
                f"Unsupported file type '{suffix}' for '{file_name}'. Please upload a .csv, .xlsx, or .xls file."
            )
        return _coerce_numeric_like_columns(df)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to load '{file_name}': {exc}") from exc