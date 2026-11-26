from pathlib import Path

from data_loader import load_dataset
from data_profiler import profile_dataset
from diagnostics_engine import run_root_cause_diagnostics
from forecast_engine import generate_metric_forecast
from kpi_engine import identify_and_compute_kpis
from qa_engine import answer_deterministic_question, build_suggested_questions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "operations_sample.csv"


def test_deterministic_qa_intents():
    df = load_dataset(SAMPLE_CSV)
    profile = profile_dataset(df)
    kpis = identify_and_compute_kpis(df, date_col="date")
    diag = run_root_cause_diagnostics(
        df, metric_col="tickets_received", date_col="date", category_cols=["department", "region"]
    )
    forecast = generate_metric_forecast(df, metric_col="tickets_received", date_col="date")

    suggestions = build_suggested_questions(df, profile, kpis)
    assert len(suggestions) >= 1

    metric_ans = answer_deterministic_question(
        "What is the total and average of tickets_received?",
        df, profile, kpis, diag, forecast,
    )
    assert metric_ans["matched_intent"] == "metric_summary"
    assert "8,498.0" in metric_ans["answer"]

    seg_ans = answer_deterministic_question(
        "Which department has the highest tickets_received?",
        df, profile, kpis, diag, forecast,
    )
    assert seg_ans["matched_intent"] == "segment_ranking"
    assert "Member Support" in seg_ans["answer"]

    fc_ans = answer_deterministic_question(
        "What is the forecast for tickets_received?",
        df, profile, kpis, diag, forecast,
    )
    assert fc_ans["matched_intent"] == "forecast"
    assert "1,367.08" in fc_ans["answer"]

    dq_ans = answer_deterministic_question(
        "Are there any duplicate rows or data quality issues?",
        df, profile, kpis, diag, forecast,
    )
    assert dq_ans["matched_intent"] == "data_quality"
    assert "18 rows" in dq_ans["answer"]