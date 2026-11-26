from __future__ import annotations

import asyncio
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from docx import Document
from docx.shared import Inches as DocxInches
from docx.shared import Pt as DocxPt
from docx.shared import RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptxRGBColor
from pptx.util import Inches, Pt

from diagnostics_engine import sanitize_label


def _strip_markdown(text: str) -> str:
    """Normalize Markdown markers and special punctuation into clean deliverable text."""
    return (
        sanitize_label(text, max_len=10000)
        .replace("`", "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("×", "x")
    )


def _add_sanitized_runs(paragraph: Any, value: str, max_len: int = 180) -> None:
    """Render sanitized text while preserving paired Markdown bold as run formatting."""
    normalized = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    parts = re.split(r"(\*\*.*?\*\*|__.*?__)", normalized)
    annotated: list[tuple[str, bool]] = []
    for part in parts:
        bold = (part.startswith("**") and part.endswith("**")) or (
            part.startswith("__") and part.endswith("__")
        )
        content = part[2:-2] if bold else part
        content = content.replace("**", "").replace("__", "")
        for char in content:
            if char.isspace():
                if annotated and annotated[-1][0] != " ":
                    annotated.append((" ", bold))
            else:
                annotated.append((char, bold))

    while annotated and annotated[0][0] == " ":
        annotated.pop(0)
    while annotated and annotated[-1][0] == " ":
        annotated.pop()

    plain_text = "".join(char for char, _ in annotated)
    clean_text = sanitize_label(plain_text, max_len=max_len, fallback="")
    if not clean_text:
        clean_text = "Unknown"
        paragraph.add_run(clean_text).bold = False
        return

    truncated = clean_text.endswith("...") and len(plain_text) > len(clean_text)
    content_length = len(clean_text) - 3 if truncated else len(clean_text)
    visible = annotated[:content_length]

    start = 0
    while start < len(visible):
        bold = visible[start][1]
        end = start + 1
        while end < len(visible) and visible[end][1] == bold:
            end += 1
        run = paragraph.add_run("".join(char for char, _ in visible[start:end]))
        run.bold = bold
        start = end
    if truncated:
        paragraph.add_run("...").bold = False


def _add_chart_image(doc: Any, caption: str, image_bytes: bytes) -> bool:
    """Save a chart temporarily and embed it only after confirming a non-empty image file."""
    caption_p = doc.add_paragraph()
    caption_run = caption_p.add_run(sanitize_label(caption))
    caption_run.bold = True
    caption_run.font.size = DocxPt(10)

    with TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "chart.png"
        image_path.write_bytes(image_bytes)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            return False
        doc.add_picture(str(image_path), width=DocxInches(6.2))
    return True


def _add_bullet_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    """Add a clean full-width title-and-bullets slide to the PowerPoint presentation."""
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = sanitize_label(title)

    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True

    clean_bullets = [
        sanitize_label(b, max_len=160)
        for b in bullets
        if b and str(b).strip()
    ]
    if not clean_bullets:
        clean_bullets = ["No items available for this section."]

    for idx, bullet in enumerate(clean_bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.space_after = Pt(10)


def _add_split_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    chart_png: bytes | None,
) -> None:
    """Add a two-column slide (bullets left, chart image right), falling back to full-width bullets."""
    if not chart_png:
        _add_bullet_slide(prs, title, bullets)
        return

    # Use Title Only layout (index 5) so we can position text box and chart cleanly
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = sanitize_label(title)

    # Left column: concise takeaway bullets
    left_box = slide.shapes.add_textbox(Inches(0.65), Inches(1.55), Inches(5.35), Inches(5.3))
    tf = left_box.text_frame
    tf.word_wrap = True

    clean_bullets = [
        sanitize_label(b, max_len=150)
        for b in bullets
        if b and str(b).strip()
    ]
    if not clean_bullets:
        clean_bullets = ["No items available for this section."]

    for idx, bullet in enumerate(clean_bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = f"•  {bullet}"
        p.font.size = Pt(14)
        p.font.color.rgb = PptxRGBColor(0x1F, 0x29, 0x37)
        p.space_after = Pt(11)

    # Right column: embedded chart image
    slide.shapes.add_picture(
        BytesIO(chart_png),
        Inches(6.25),
        Inches(1.65),
        width=Inches(6.45),
    )


def _render_plotly_png(
    fig: Any,
    width: int = 900,
    height: int = 460,
) -> bytes | None:
    """Render a Plotly figure to PNG bytes in memory with clean light-theme corporate styling."""
    if fig is None:
        return None
    try:
        import plotly.graph_objects as go

        clean_fig = go.Figure(fig.to_dict())
        clean_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=["#1D4ED8", "#0D9488", "#D97706", "#6366F1"],
            font=dict(color="#1F2937", size=12),
            margin=dict(l=50, r=25, t=55, b=45),
        )
        # Ensure primary bars/lines don't stay stuck on black from dark-mode defaults
        for trace in clean_fig.data:
            if getattr(trace, "type", "") == "bar":
                if not getattr(trace.marker, "color", None):
                    trace.marker.color = "#1D4ED8"
            elif getattr(trace, "type", "") in ("scatter", "scattergl"):
                if not getattr(trace.line, "color", None):
                    trace.line.color = "#1D4ED8"
                if not getattr(trace.marker, "color", None):
                    trace.marker.color = "#1D4ED8"

        try:
            img_bytes = clean_fig.to_image(format="png", width=width, height=height, scale=2)
            if img_bytes:
                return img_bytes
        except Exception:
            import kaleido

            with TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "chart.png"
                asyncio.run(
                    kaleido.write_fig(
                        clean_fig,
                        path=image_path,
                        opts={"format": "png", "width": width, "height": height, "scale": 2},
                        cancel_on_error=True,
                    )
                )
                if image_path.is_file() and image_path.stat().st_size:
                    return image_path.read_bytes()
    except Exception:
        return None
    return None


def _build_docx_brief(
    clean_source: str,
    generated_at: str,
    focus_metric: str | None,
    kpi_result: dict[str, Any],
    headline_bullets: list[str],
    driver_bullets: list[str],
    forecast_bullets: list[str],
    caveat_bullets: list[str],
    trend_png: bytes | None = None,
    breakdown_png: bytes | None = None,
) -> bytes:
    """Build a formatted Word (.docx) Executive Brief in memory with up to two key charts."""
    doc = Document()

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = DocxPt(10.5)
    normal_style.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)

    title = doc.add_heading(f"Executive Brief: {clean_source}", level=0)
    title.runs[0].font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    meta_parts = [f"Generated by Danica ({generated_at})"]
    if focus_metric:
        meta_parts.append(f"Primary Focus Metric: {focus_metric.replace('_', ' ').title()}")
    meta_p = doc.add_paragraph("  |  ".join(meta_parts))
    meta_p.runs[0].font.size = DocxPt(9.5)
    meta_p.runs[0].font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    # 1. Executive Highlights
    doc.add_heading("1. Executive Highlights", level=1)
    for bullet in headline_bullets:
        p = doc.add_paragraph(style="List Bullet")
        _add_sanitized_runs(p, bullet)
        p.paragraph_format.space_after = DocxPt(4)

    # 2. KPI Summary Table
    kpis = kpi_result.get("kpis", [])
    if kpis:
        doc.add_heading("2. Key Performance Indicators", level=1)
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        headers = ["Metric", "Aggregation", "Current Value", "Prior-Period Change", "Range (Min - Max)"]
        for idx, text in enumerate(headers):
            cell = table.rows[0].cells[idx]
            cell.text = text
            for run in cell.paragraphs[0].runs:
                run.bold = True

        for kpi in kpis:
            row_cells = table.add_row().cells
            delta_str = "N/A"
            if kpi.get("delta") and kpi["delta"].get("pct_change") is not None:
                delta_str = f"{kpi['delta']['pct_change']:+.1f}%"
            row_cells[0].text = sanitize_label(kpi["label"])
            row_cells[1].text = sanitize_label(kpi["aggregation"])
            row_cells[2].text = f"{kpi['primary_value']:,}"
            row_cells[3].text = delta_str
            row_cells[4].text = f"{kpi['min']:,} - {kpi['max']:,}"

        doc.add_paragraph()

    # 3. Key Visuals (up to 2 charts: trend + segment breakdown)
    doc.add_heading("3. Visual Performance Snapshot", level=1)
    if trend_png:
        if not _add_chart_image(doc, "Performance Trend Over Time", trend_png):
            doc.add_paragraph("Performance trend chart could not be saved; no image was embedded.")
    else:
        doc.add_paragraph("Performance trend chart is unavailable for this dataset.")

    if breakdown_png:
        if not _add_chart_image(doc, "Top Segment Breakdown", breakdown_png):
            doc.add_paragraph("Top Segment Breakdown chart could not be saved; no image was embedded.")
    else:
        doc.add_paragraph("Top Segment Breakdown chart is unavailable for this dataset.")

    # 4. Root-Cause & Segment Drivers
    if driver_bullets:
        doc.add_heading("4. Root-Cause & Segment Drivers", level=1)
        for bullet in driver_bullets:
            p = doc.add_paragraph(style="List Bullet")
            _add_sanitized_runs(p, bullet)
            p.paragraph_format.space_after = DocxPt(4)

    # 5. Short-Term Outlook
    if forecast_bullets:
        doc.add_heading("5. Short-Term Outlook", level=1)
        for bullet in forecast_bullets:
            p = doc.add_paragraph(style="List Bullet")
            _add_sanitized_runs(p, bullet)
            p.paragraph_format.space_after = DocxPt(4)

    # 6. Data Readiness & Analyst Caveats
    doc.add_heading("6. Data Readiness & Analyst Caveats", level=1)
    for caveat in caveat_bullets or ["No critical data readiness caveats flagged."]:
        p = doc.add_paragraph(style="List Bullet")
        _add_sanitized_runs(p, caveat)
        p.paragraph_format.space_after = DocxPt(4)

    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    return docx_buffer.getvalue()


def build_executive_brief(
    source_name: str,
    profile: dict[str, Any],
    kpi_result: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    forecast_result: dict[str, Any] | None = None,
    chart_bundle: dict[str, Any] | None = None,
    focus_metric: str | None = None,
) -> dict[str, Any]:
    """Assemble structured executive summary bullets, a .docx brief, and a visual .pptx stakeholder deck."""
    clean_source = sanitize_label(source_name or "dataset", max_len=60, fallback="dataset")

    headline_bullets: list[str] = [
        f"Dataset {clean_source} contains {profile['row_count']:,} rows across {profile['col_count']} columns ({profile['duplicate_rows']} duplicate rows)."
    ]

    for kpi in kpi_result.get("kpis", [])[:4]:
        delta_txt = ""
        if kpi.get("delta") and kpi["delta"].get("pct_change") is not None:
            delta_txt = f" ({kpi['delta']['pct_change']:+.1f}% vs. prior period)"
        headline_bullets.append(
            f"{sanitize_label(kpi['label'])} ({sanitize_label(kpi['aggregation'])}): "
            f"{kpi['primary_value']:,}{delta_txt}."
        )

    driver_bullets: list[str] = []
    if diagnostics:
        driver_bullets.extend(
            sanitize_label(item, max_len=180)
            for item in diagnostics.get("findings", [])[:4]
        )

    forecast_bullets: list[str] = []
    if forecast_result:
        if forecast_result.get("available"):
            metric_label = forecast_result["metric"].replace("_", " ").title()
            forecast_bullets.append(
                f"{metric_label} latest actual is {forecast_result['last_actual']:,}, projected to {forecast_result['next_forecast']:,} next period via {forecast_result['method']}."
            )
        else:
            forecast_bullets.append(
                sanitize_label(forecast_result.get("reason", "Forecast not available."), max_len=180)
            )

    caveat_bullets: list[str] = []
    caveat_bullets.extend(profile.get("readiness_notes", []))
    caveat_bullets.extend(kpi_result.get("notes", []))
    if forecast_result:
        for note in forecast_result.get("notes", []):
            if "Caution:" in note:
                caveat_bullets.append(note)

    seen: set[str] = set()
    unique_caveats: list[str] = []
    for item in caveat_bullets:
        cleaned = sanitize_label(item, max_len=180, fallback="")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_caveats.append(cleaned)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Render charts once so both .docx and .pptx share clean visuals
    trend_png = _render_plotly_png(chart_bundle.get("trend_fig")) if chart_bundle else None
    breakdown_png = _render_plotly_png(chart_bundle.get("breakdown_fig")) if chart_bundle else None
    forecast_png = (
        _render_plotly_png(forecast_result.get("chart"))
        if forecast_result and forecast_result.get("available")
        else None
    )

    # Build Visual Stakeholder PowerPoint (.pptx)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = sanitize_label(
        f"Stakeholder Deck: {clean_source}",
        max_len=60,
    )
    if len(title_slide.placeholders) > 1:
        title_slide.placeholders[1].text = (
            sanitize_label(
                f"Generated by Danica - Data Analyst Assistant ({generated_at})",
                max_len=120,
            )
        )

    _add_split_slide(prs, "Key Performance Highlights", headline_bullets, trend_png)
    if driver_bullets:
        _add_split_slide(prs, "Root-Cause & Segment Drivers", driver_bullets, breakdown_png)
    if forecast_bullets:
        _add_split_slide(prs, "Short-Term Outlook", forecast_bullets, forecast_png)
    _add_bullet_slide(prs, "Data Readiness & Analyst Caveats", unique_caveats)

    pptx_buffer = BytesIO()
    prs.save(pptx_buffer)
    pptx_bytes = pptx_buffer.getvalue()

    # Build Executive Brief Word Document (.docx)
    docx_bytes = _build_docx_brief(
        clean_source=clean_source,
        generated_at=generated_at,
        focus_metric=focus_metric,
        kpi_result=kpi_result,
        headline_bullets=headline_bullets,
        driver_bullets=driver_bullets,
        forecast_bullets=forecast_bullets,
        caveat_bullets=unique_caveats,
        trend_png=trend_png,
        breakdown_png=breakdown_png,
    )

    return {
        "source_name": clean_source,
        "generated_at": generated_at,
        "headline_bullets": [sanitize_label(b, max_len=180) for b in headline_bullets],
        "driver_bullets": [sanitize_label(b, max_len=180) for b in driver_bullets],
        "forecast_bullets": [sanitize_label(b, max_len=180) for b in forecast_bullets],
        "caveat_bullets": unique_caveats,
        "docx_bytes": docx_bytes,
        "pptx_bytes": pptx_bytes,
    }
