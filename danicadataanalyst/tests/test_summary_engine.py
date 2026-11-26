from io import BytesIO
from pathlib import Path

from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from chart_engine import generate_milestone3_charts
from data_loader import load_dataset
from data_profiler import profile_dataset
from diagnostics_engine import run_root_cause_diagnostics
from forecast_engine import generate_metric_forecast
from kpi_engine import identify_and_compute_kpis
from summary_engine import _add_sanitized_runs, build_executive_brief

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def _count_pictures(slide) -> int:
    return sum(1 for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE)


def test_build_executive_brief_docx_and_pptx_end_to_end():
    df = load_dataset(SAMPLE_CSV)
    profile = profile_dataset(df)
    kpis = identify_and_compute_kpis(df, date_col="date")
    charts = generate_milestone3_charts(
        df, metric_col="tickets_received", date_col="date", category_col="department"
    )
    diag = run_root_cause_diagnostics(
        df, metric_col="tickets_received", date_col="date", category_cols=["department", "region"]
    )
    forecast = generate_metric_forecast(df, metric_col="tickets_received", date_col="date")

    brief = build_executive_brief(
        source_name="operations_sample.csv",
        profile=profile,
        kpi_result=kpis,
        diagnostics=diag,
        forecast_result=forecast,
        chart_bundle=charts,
        focus_metric="tickets_received",
    )

    assert len(brief["headline_bullets"]) >= 2
    assert len(brief["driver_bullets"]) >= 1
    assert len(brief["forecast_bullets"]) == 1

    # Validate Word (.docx) Executive Brief
    assert isinstance(brief["docx_bytes"], bytes)
    assert len(brief["docx_bytes"]) > 2000
    doc = Document(BytesIO(brief["docx_bytes"]))
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Executive Brief: operations_sample.csv" in doc_text
    assert "Visual Performance Snapshot" in doc_text
    assert len(doc.tables) == 1
    assert len(doc.inline_shapes) == 2

    # Validate Visual PowerPoint (.pptx) Stakeholder Deck
    assert isinstance(brief["pptx_bytes"], bytes)
    assert len(brief["pptx_bytes"]) > 2000
    prs = Presentation(BytesIO(brief["pptx_bytes"]))
    assert len(prs.slides) == 5
    assert prs.slides[0].shapes.title.text == "Stakeholder Deck: operations_sample.csv"
    assert prs.slides[1].shapes.title.text == "Key Performance Highlights"
    assert _count_pictures(prs.slides[1]) == 1
    assert prs.slides[2].shapes.title.text == "Root-Cause & Segment Drivers"
    assert _count_pictures(prs.slides[2]) == 1
    assert prs.slides[3].shapes.title.text == "Short-Term Outlook"
    assert _count_pictures(prs.slides[3]) == 1
    assert prs.slides[4].shapes.title.text == "Data Readiness & Analyst Caveats"


def test_sanitized_docx_runs_and_section_three_fallback():
    doc = Document()
    paragraph = doc.add_paragraph()
    _add_sanitized_runs(paragraph, "**First\n\nline** and __second line__", max_len=60)
    assert paragraph.text == "First line and second line"
    assert paragraph.runs[0].bold is True
    assert paragraph.runs[-1].bold is True

    brief = build_executive_brief(
        source_name="dataset.csv",
        profile={
            "row_count": 1,
            "col_count": 1,
            "duplicate_rows": 0,
            "readiness_notes": [],
        },
        kpi_result={"kpis": [], "notes": []},
        diagnostics={"findings": ["**Long\n\nprose** " + "detail " * 60]},
        chart_bundle=None,
    )
    output_doc = Document(BytesIO(brief["docx_bytes"]))
    text = "\n".join(paragraph.text for paragraph in output_doc.paragraphs)
    assert "**" not in text
    assert "\n\n" not in text
    assert "Top Segment Breakdown chart is unavailable" in text
    assert len(output_doc.inline_shapes) == 0
