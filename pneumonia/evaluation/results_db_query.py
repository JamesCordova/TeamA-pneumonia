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

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from pneumonia.data.load_data import get_db_engine
from pneumonia.evaluation.metrics import compute_all_metrics


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


def read_predictions_from_db(
    department: str, age_group: str, measure: str = "cases", database_url: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    DB equivalent of pneumonia.visualization._utils.read_predictions(): shaped
    like the local *_predictions.csv concat (date, split, model, actual,
    predicted, horizon, department, age_group) — sourced from results_unsa_ira's
    latest run per run_name. Every row has split='backtest'; the database only
    stores walk-forward results, not the classic train/val/test predictions
    (see pneumonia.evaluation.results_db).

    When horizon > step, overlapping steps forecast the same date at different
    horizon_offsets — kept to the smallest offset per (run_name, date), the
    same tie-break pneumonia.visualization.persistence.save_walkforward_predictions
    uses for the local CSV, so a given model only ever draws one line.
    """
    runs = latest_runs(department, age_group, measure, database_url)
    if runs.empty:
        return None
    preds = predictions_for_runs(runs["run_id"].tolist(), database_url)
    if preds.empty:
        return None
    preds = preds.merge(runs[["run_id", "run_name"]], on="run_id", how="left")
    preds = (
        preds.sort_values("horizon_offset")
             .drop_duplicates(subset=["run_name", "date"], keep="first")
    )
    return pd.DataFrame({
        "date":       preds["date"],
        "split":      "backtest",
        "model":      preds["run_name"],
        "actual":     preds["actual"],
        "predicted":  preds["predicted"],
        "horizon":    preds["horizon_offset"],
        "department": department,
        "age_group":  age_group,
    })


def reconstruct_results_from_run(run_id: int, database_url: Optional[str] = None) -> dict:
    """
    Rebuild a WalkForwardValidator.run()-shaped dict (config, model_params,
    n_steps, step_results, predictions, metrics_by_horizon) from an existing
    run's stored walkforward_predictions — so scripts/run_walkforward.py can
    regenerate the local *_predictions.csv / *_step_metrics.csv / JSON files
    for a configuration that was already run, without retraining.

    step_results entries won't have 'train_start'/'train_end'/'n_train' (not
    persisted in walkforward_predictions) — set to None. Harmless: neither
    pneumonia.visualization.persistence.save_step_metrics nor
    save_walkforward_predictions reads those fields.
    """
    engine = get_db_engine(database_url)
    with engine.connect() as conn:
        run_row = conn.execute(
            text("""
                SELECT model, run_name, horizon, step, window_type, train_size,
                       refit_every, model_params, n_steps
                FROM results_unsa_ira.walkforward_runs
                WHERE run_id = :run_id
            """),
            {"run_id": run_id},
        ).mappings().first()
        if run_row is None:
            raise ValueError(f"run_id={run_id} not found in results_unsa_ira.walkforward_runs")

        preds = pd.read_sql(
            text("""
                SELECT step_idx, horizon_offset, date, actual, predicted
                FROM results_unsa_ira.walkforward_predictions
                WHERE run_id = :run_id
            """),
            conn, params={"run_id": run_id}, parse_dates=["date"],
        )

    horizon = run_row["horizon"]
    config = {
        "horizon":     horizon,
        "step":        run_row["step"],
        "window_type": run_row["window_type"],
        "train_size":  run_row["train_size"],
        "refit_every": run_row["refit_every"],
    }

    metrics_by_horizon = {}
    for h, g in preds.groupby("horizon_offset"):
        metrics_by_horizon[int(h)] = compute_all_metrics(
            g["actual"].values, g["predicted"].values, warn_on_nan=False
        )

    step_results = []
    for step_idx, g in preds.groupby("step_idx"):
        g = g.sort_values("horizon_offset")
        step_results.append({
            "step":           int(step_idx),
            "train_start":    None,
            "train_end":      None,
            "forecast_start": str(g["date"].min().date()),
            "forecast_end":   str(g["date"].max().date()),
            "n_train":        None,
            "dates":          [str(d.date()) for d in g["date"]],
            "actuals":        g["actual"].tolist(),
            "predictions":    g["predicted"].tolist(),
            "metrics": compute_all_metrics(
                g["actual"].values, g["predicted"].values, warn_on_nan=False
            ),
        })
    step_results.sort(key=lambda s: s["step"])

    # Wide predictions DataFrame: one row per date, one column per horizon
    # offset (pred_h1..pred_hN) — the same shape WalkForwardValidator.run()
    # builds directly. A date can have several offsets filled in (from
    # different overlapping steps when horizon > step); each (date,
    # horizon_offset) pair is unique by construction (one step per offset).
    wide = preds.pivot_table(
        index="date", columns="horizon_offset", values="predicted", aggfunc="first"
    )
    wide = wide.rename(columns=lambda h: f"pred_h{int(h)}")
    for h in range(1, horizon + 1):
        col = f"pred_h{h}"
        if col not in wide.columns:
            wide[col] = np.nan

    actual_by_date = preds.groupby("date")["actual"].first()
    pred_df = wide.sort_index()
    pred_df.insert(0, "actual", actual_by_date.reindex(pred_df.index))

    return {
        "config":             config,
        "model_params":       run_row["model_params"],
        "n_steps":            run_row["n_steps"],
        "step_results":       step_results,
        "predictions":        pred_df,
        "metrics_by_horizon": metrics_by_horizon,
    }
