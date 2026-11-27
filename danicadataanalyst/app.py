import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

import qa_engine
import summary_engine
from chart_engine import generate_milestone3_charts
from data_loader import DataLoadError, load_dataset
from data_profiler import profile_dataset, synthesize_calendar_date
from diagnostics_engine import (
    format_column_label,
    is_free_text_column,
    is_identifier_column,
    run_root_cause_diagnostics,
)
from forecast_engine import generate_metric_forecast
from kpi_engine import identify_and_compute_kpis

st.set_page_config(
    page_title="Danica — Data Analyst Assistant",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "operations_sample.csv"

st.title("Danica — Data Analyst Assistant")
st.caption("Milestone 6 — Executive Brief (.docx) & Stakeholder Deck (.pptx) Export")

st.sidebar.header("1. Data Source")
use_sample = st.sidebar.checkbox(
    "Use sample operations dataset",
    value=SAMPLE_DATA_PATH.exists(),
)
uploaded_file = st.sidebar.file_uploader(
    "Or upload CSV / Excel",
    type=["csv", "xlsx", "xls"],
)

df: pd.DataFrame | None = None
source_label = ""

try:
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        source_label = uploaded_file.name
    elif use_sample and SAMPLE_DATA_PATH.exists():
        df = load_dataset(SAMPLE_DATA_PATH)
        source_label = SAMPLE_DATA_PATH.name
except DataLoadError as exc:
    st.error(str(exc))

if df is None:
    st.info("Select the sample dataset or upload a CSV/Excel file in the sidebar to begin.")
    st.stop()

profile = profile_dataset(df)
derived_date_col = synthesize_calendar_date(df, profile["datetime_columns"])
if derived_date_col:
    profile["datetime_columns"].append(derived_date_col)
    profile["readiness_notes"] = [
        note
        for note in profile["readiness_notes"]
        if not note.startswith("No date/time column detected:")
    ]
default_date_col = profile["datetime_columns"][0] if profile["datetime_columns"] else None
kpi_result = identify_and_compute_kpis(df, date_col=default_date_col)

st.session_state["active_df"] = df
st.session_state["active_profile"] = profile
st.session_state["active_kpis"] = kpi_result

st.subheader(f"Dataset Overview: `{source_label}`")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", f"{profile['row_count']:,}")
col2.metric("Columns", profile["col_count"])
col3.metric("Numeric Cols", len(profile["numeric_columns"]))
col4.metric("Boolean Flags", len(profile["boolean_columns"]))
col5.metric("Duplicate Rows", profile["duplicate_rows"])

st.markdown("### Key Performance Indicators (KPIs)")
if kpi_result["kpis"]:
    kpi_cols = st.columns(min(4, len(kpi_result["kpis"])))
    for idx, kpi in enumerate(kpi_result["kpis"][:4]):
        delta_str = None
        if kpi["delta"] and kpi["delta"]["pct_change"] is not None:
            delta_str = f"{kpi['delta']['pct_change']:+.1f}% ({kpi['delta']['comparison']})"
        kpi_cols[idx].metric(
            label=f"{kpi['label']} ({kpi['aggregation']})",
            value=f"{kpi['primary_value']:,}",
            delta=delta_str,
        )

    kpi_rows = []
    for kpi in kpi_result["kpis"]:
        kpi_rows.append(
            {
                "Metric": kpi["label"],
                "Primary Agg": kpi["aggregation"],
                "Primary Value": kpi["primary_value"],
                "Mean": kpi["mean"],
                "Median": kpi["median"],
                "Min": kpi["min"],
                "Max": kpi["max"],
                "Period Delta": (
                    f"{kpi['delta']['pct_change']:+.1f}%"
                    if kpi["delta"] and kpi["delta"]["pct_change"] is not None
                    else "N/A"
                ),
            }
        )
    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)
else:
    st.info("No numeric KPIs available for this dataset.")

st.markdown("### Interactive Charts & Visuals")
available_metrics = [kpi["metric"] for kpi in kpi_result["kpis"]]
categorical_cols = [
    col_info["column"]
    for col_info in profile["columns"]
    if col_info["type"] == "categorical"
    and not is_free_text_column(df[col_info["column"]])
    and not is_identifier_column(df[col_info["column"]], col_info["column"])
]

selected_metric = None
chart_bundle = None
diagnostics = None
forecast_result = None

if available_metrics:
    ctrl_col1, ctrl_col2 = st.columns(2)
    selected_metric = ctrl_col1.selectbox("Select KPI Metric to Visualize", available_metrics)
    selected_category = (
        ctrl_col2.selectbox("Breakdown Dimension", categorical_cols)
        if categorical_cols
        else None
    )

    chart_bundle = generate_milestone3_charts(
        df=df,
        metric_col=selected_metric,
        date_col=default_date_col,
        category_col=selected_category,
    )

    st.session_state["active_charts"] = chart_bundle

    if chart_bundle["trend_fig"] is not None:
        st.plotly_chart(chart_bundle["trend_fig"], use_container_width=True)

    viz_col1, viz_col2 = st.columns(2)
    if chart_bundle["breakdown_fig"] is not None:
        viz_col1.plotly_chart(chart_bundle["breakdown_fig"], use_container_width=True)
    if chart_bundle["distribution_fig"] is not None:
        viz_col2.plotly_chart(chart_bundle["distribution_fig"], use_container_width=True)

    for note in chart_bundle["notes"]:
        st.caption(f"Chart Note: {note}")

    st.markdown(
        f"### Root-Cause & Driver Diagnostics (`{format_column_label(selected_metric)}`)"
    )
    diagnostics = run_root_cause_diagnostics(
        df=df,
        metric_col=selected_metric,
        date_col=default_date_col,
        category_cols=categorical_cols,
    )
    st.session_state["active_diagnostics"] = diagnostics

    for finding in diagnostics["findings"]:
        st.write(f"- {finding}")

    if diagnostics["segment_drivers"]:
        st.markdown("**Top Segment Drivers**")
        driver_table = pd.DataFrame(diagnostics["segment_drivers"])
        st.dataframe(
            driver_table,
            column_config={
                "dimension": st.column_config.TextColumn("Dimension", width="medium"),
                "top_segment": st.column_config.TextColumn("Top Segment", width="medium"),
                "impact_metric": st.column_config.TextColumn("Impact Metric", width="large"),
            },
            use_container_width=True,
        )
    if diagnostics["period_movers"]:
        st.markdown("**Largest Period-over-Period Segment Shifts**")
        st.dataframe(
            pd.DataFrame(diagnostics["period_movers"]),
            column_config={
                "dimension": st.column_config.TextColumn("Dimension", width="medium"),
                "segment": st.column_config.TextColumn("Segment", width="medium"),
                "comparison": st.column_config.TextColumn("Comparison", width="medium"),
            },
            use_container_width=True,
        )

    if not diagnostics["outliers"].empty:
        st.markdown("**Flagged Outlier Records (1.5×IQR)**")
        st.dataframe(diagnostics["outliers"], use_container_width=True)

    st.markdown(f"### Short-Term Forecast (`{format_column_label(selected_metric)}`)")
    forecast_result = generate_metric_forecast(
        df=df,
        metric_col=selected_metric,
        date_col=default_date_col,
        periods_ahead=3,
    )
    st.session_state["active_forecast"] = forecast_result

    if forecast_result["available"]:
        f_col1, f_col2 = st.columns(2)
        f_col1.metric("Latest Actual Period", f"{forecast_result['last_actual']:,}")
        f_col2.metric("Next Period Forecast", f"{forecast_result['next_forecast']:,}")
        st.plotly_chart(forecast_result["chart"], use_container_width=True)
        st.dataframe(forecast_result["forecast_df"], use_container_width=True)
    else:
        st.info(forecast_result["reason"])

    for note in forecast_result["notes"]:
        st.caption(f"Forecast Note: {note}")
else:
    st.info("No numeric metrics available to generate charts, diagnostics, or forecasts.")

importlib.reload(qa_engine)

st.markdown("### Guided Analytics Q&A (Deterministic)")
st.caption("Rule-based dataset lookup using pandas and Danica's computed metrics—no external AI API.")

suggestions = qa_engine.build_suggested_questions(df, profile, kpi_result)
if suggestions:
    st.markdown("**Suggested Questions:** " + " • ".join(f"`{q}`" for q in suggestions))

user_question = st.text_input(
    "Ask a question about this dataset",
    placeholder="e.g., Which department has the highest tickets_received?",
)
if user_question:
    qa_out = qa_engine.answer_deterministic_question(
        question=user_question,
        df=df,
        profile=profile,
        kpi_result=kpi_result,
        diagnostics=diagnostics,
        forecast_result=forecast_result,
    )
    st.info(qa_out["answer"])
    if qa_out.get("table") is not None and not qa_out["table"].empty:
        st.dataframe(qa_out["table"], use_container_width=True)

importlib.reload(summary_engine)

st.markdown("### Executive Brief & Deliverable Exports")
brief = summary_engine.build_executive_brief(
    source_name=source_label,
    profile=profile,
    kpi_result=kpi_result,
    diagnostics=diagnostics,
    forecast_result=forecast_result,
    chart_bundle=chart_bundle,
    focus_metric=selected_metric,
)
st.session_state["active_brief"] = brief

for bullet in brief["headline_bullets"] + brief["driver_bullets"] + brief["forecast_bullets"]:
    st.write(f"- {bullet}")

export_stem = Path(source_label).stem if source_label else "danica_report"
dl_col1, dl_col2 = st.columns(2)

docx_data = brief.get("docx_bytes")
if docx_data:
    dl_col1.download_button(
        label="Download Executive Brief (.docx)",
        data=docx_data,
        file_name=f"{export_stem}_executive_brief.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

pptx_data = brief.get("pptx_bytes")
if pptx_data:
    dl_col2.download_button(
        label="Download Stakeholder Deck (.pptx)",
        data=pptx_data,
        file_name=f"{export_stem}_stakeholder_deck.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

st.markdown("### Analyst Notes (Data Readiness & Metric Caveats)")
for note in profile["readiness_notes"] + kpi_result["notes"]:
    st.write(f"- {note}")

st.markdown("### Column Summary")
column_summary = pd.DataFrame(profile["columns"])
if "column" in column_summary.columns:
    column_summary["column"] = column_summary["column"].map(format_column_label)
column_summary_config = {}
for col in column_summary.columns:
    label = format_column_label(col)
    if col == "missing_count":
        column_summary_config[col] = st.column_config.NumberColumn(label, width="medium")
    else:
        width = "large" if col == "column" else "medium"
        column_summary_config[col] = st.column_config.TextColumn(label, width=width)
st.dataframe(
    column_summary,
    column_config=column_summary_config,
    use_container_width=True,
)

if profile["numeric_summary"]:
    st.markdown("### Numeric Summary")
    numeric_summary = pd.DataFrame(profile["numeric_summary"]).T
    numeric_summary.index = [
        format_column_label(col) for col in numeric_summary.index
    ]
    numeric_summary = numeric_summary.rename(
        columns={
            "min": "Min",
            "median": "Median",
            "mean": "Mean",
            "max": "Max",
        }
    )
    st.dataframe(
        numeric_summary,
        column_config={
            col: st.column_config.NumberColumn(col, width="medium")
            for col in numeric_summary.columns
        },
        use_container_width=True,
    )

st.markdown("### Data Preview (First 15 Rows)")
preview = df.head(15)
preview_column_config = {}
for col in preview.columns:
    label = format_column_label(col)
    width = (
        "large"
        if len(label) > 16 or is_identifier_column(df[col], str(col))
        else "medium"
    )
    if pd.api.types.is_numeric_dtype(df[col]):
        preview_column_config[col] = st.column_config.NumberColumn(label, width=width)
    else:
        preview_column_config[col] = st.column_config.TextColumn(label, width=width)
st.dataframe(
    preview,
    column_order=list(preview.columns),
    column_config=preview_column_config,
    use_container_width=True,
)
