"""
Read WalkForwardValidator results back from results_unsa_ira
(db/migrations/0001_walkforward_results) — the counterpart to
pneumonia.evaluation.results_db's writer.

Only two primitives are provided here: which runs exist, and the raw
predictions for them. All aggregation (by horizon, by step, pooled, or as a
cumulative window) happens in scripts/compare_models.py by grouping the raw
rows with pandas and pneumonia.evaluation.metrics.compute_all_metrics — the
same function used everywhere else, so DB-sourced and file-sourced metrics are
computed identically.
"""

from typing import Optional, Sequence

import pandas as pd
from sqlalchemy import bindparam, text

from pneumonia.data.load_data import get_db_engine


def latest_runs(
    department: str,
    age_group: str,
    measure: str = "cases",
    database_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    One row per run_name: only its most recent run (by created_at). A run_name
    can have many rows in walkforward_runs (reruns, grid-search trials) since
    rows are never overwritten — this picks the latest one for comparison,
    the same way each *_walkforward_metrics.json file only ever reflected the
    last run for that name.

    Returns columns: run_id, run_name, model, horizon, step, window_type,
    train_size, refit_every, model_params, n_steps, created_at.
    """
    engine = get_db_engine(database_url)
    query = text("""
        SELECT DISTINCT ON (run_name)
            run_id, run_name, model, horizon, step, window_type,
            train_size, refit_every, model_params, n_steps, created_at
        FROM results_unsa_ira.walkforward_runs
        WHERE department = :department AND age_group = :age_group AND measure = :measure
        ORDER BY run_name, created_at DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn,
            params={"department": department, "age_group": age_group, "measure": measure},
        )
    return df


def predictions_for_runs(
    run_ids: Sequence[int], database_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Raw predictions for the given run_ids — every row, not deduplicated by
    date (overlapping steps, when horizon > step, keep all their rows).

    Returns columns: run_id, step_idx, horizon_offset, date, actual, predicted.
    """
    if not run_ids:
        return pd.DataFrame(
            columns=["run_id", "step_idx", "horizon_offset", "date", "actual", "predicted"]
        )
    engine = get_db_engine(database_url)
    stmt = text("""
        SELECT run_id, step_idx, horizon_offset, date, actual, predicted
        FROM results_unsa_ira.walkforward_predictions
        WHERE run_id IN :run_ids
    """).bindparams(bindparam("run_ids", expanding=True))
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn, params={"run_ids": list(run_ids)}, parse_dates=["date"])
    return df
