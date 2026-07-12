"""
Persist WalkForwardValidator results to the results_unsa_ira schema
(db/migrations/0001_walkforward_results) — additive to the existing
*_walkforward_metrics.json / *_step_metrics.csv / *_predictions.csv files,
not a replacement for them.

Every call inserts a new row into walkforward_runs (never overwritten, unlike
the JSON files), plus one row per individual forecast into
walkforward_predictions — raw actual/predicted, not deduplicated by date, so
overlapping steps (horizon > step) are all preserved. Metrics are computed
on demand from these raw rows, never stored pre-aggregated.
"""

import json
from typing import Optional

from sqlalchemy import text

from pneumonia.data.load_data import get_db_engine
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def save_walkforward_run(
    department: str,
    age_group: str,
    model: str,
    run_name: str,
    config: dict,
    model_params: dict,
    n_steps: int,
    step_results: list,
    measure: str = "cases",
    database_url: Optional[str] = None,
) -> int:
    """
    Insert one walkforward_runs row plus its walkforward_predictions rows.

    Args:
        department:    Department name (uppercase).
        age_group:     'under5' or '60plus'.
        model:         Model class name (e.g. 'XGBoost').
        run_name:      Run label (defaults to model name; distinguishes
                       hyperparameter variants, e.g. 'SARIMAv1.1').
        config:        `WalkForwardValidator.run()["config"]` — horizon, step,
                       window_type, train_size, refit_every.
        model_params:  `WalkForwardValidator.run()["model_params"]`.
        n_steps:       `WalkForwardValidator.run()["n_steps"]`.
        step_results:  `WalkForwardValidator.run()["step_results"]` — each dict
                       must include 'step', 'dates', 'actuals', 'predictions'.
        measure:       Target variable forecast (default 'cases' — the only
                       one get_departmental_data() supports today).
        database_url:  Overrides config.DATABASE_URL if given.

    Returns:
        The new run_id.
    """
    engine = get_db_engine(database_url)

    with engine.begin() as conn:
        run_id = conn.execute(
            text("""
                INSERT INTO results_unsa_ira.walkforward_runs
                    (department, age_group, measure, model, run_name,
                     horizon, step, window_type, train_size, refit_every,
                     model_params, n_steps)
                VALUES
                    (:department, :age_group, :measure, :model, :run_name,
                     :horizon, :step, :window_type, :train_size, :refit_every,
                     :model_params, :n_steps)
                RETURNING run_id
            """),
            {
                "department": department,
                "age_group": age_group,
                "measure": measure,
                "model": model,
                "run_name": run_name,
                "horizon": config["horizon"],
                "step": config["step"],
                "window_type": config["window_type"],
                "train_size": config["train_size"],
                "refit_every": config["refit_every"],
                "model_params": json.dumps(model_params, default=str),
                "n_steps": n_steps,
            },
        ).scalar_one()

        rows = [
            {
                "run_id": run_id,
                "step_idx": step["step"],
                "horizon_offset": j + 1,
                "date": date,
                "actual": actual,
                "predicted": predicted,
            }
            for step in step_results
            for j, (date, actual, predicted) in enumerate(
                zip(step["dates"], step["actuals"], step["predictions"])
            )
        ]
        if rows:
            conn.execute(
                text("""
                    INSERT INTO results_unsa_ira.walkforward_predictions
                        (run_id, step_idx, horizon_offset, date, actual, predicted)
                    VALUES
                        (:run_id, :step_idx, :horizon_offset, :date, :actual, :predicted)
                """),
                rows,
            )

    logger.info(
        f"Saved run_id={run_id} to results_unsa_ira "
        f"({department}/{age_group}/{measure}/{model}, {len(rows)} predictions)"
    )
    return run_id
