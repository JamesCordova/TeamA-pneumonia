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


_MATCH_CONFIG_QUERY = """
    SELECT run_id, run_name FROM results_unsa_ira.walkforward_runs
    WHERE department = :department AND age_group = :age_group AND measure = :measure
      AND model = :model AND horizon = :horizon AND step = :step
      AND window_type = :window_type AND train_size = :train_size
      AND refit_every = :refit_every
      AND model_params = CAST(:model_params AS jsonb)
"""


def _resolve_run(conn, department, age_group, measure, model, run_name, config, model_params_json):
    """
    Return (existing_run_id_or_None, resolved_run_name) for this config/run_name
    within one connection — shared by save_walkforward_run() (under an advisory
    lock, for the real insert) and resolve_run_name() (read-only preview).
    """
    existing = conn.execute(
        text(_MATCH_CONFIG_QUERY),
        {
            "department": department,
            "age_group": age_group,
            "measure": measure,
            "model": model,
            "horizon": config["horizon"],
            "step": config["step"],
            "window_type": config["window_type"],
            "train_size": config["train_size"],
            "refit_every": config["refit_every"],
            "model_params": model_params_json,
        },
    ).first()
    if existing is not None:
        return existing.run_id, existing.run_name

    existing_names = set(
        conn.execute(
            text("""
                SELECT run_name FROM results_unsa_ira.walkforward_runs
                WHERE department = :department AND age_group = :age_group AND measure = :measure
            """),
            {"department": department, "age_group": age_group, "measure": measure},
        ).scalars().all()
    )
    resolved_name = run_name
    if resolved_name in existing_names:
        n = 1
        while f"{run_name}({n})" in existing_names:
            n += 1
        resolved_name = f"{run_name}({n})"
        logger.info(
            f"run_name '{run_name}' already used by a different configuration — "
            f"saving this one as '{resolved_name}'."
        )
    return None, resolved_name


def resolve_run_name(
    department: str,
    age_group: str,
    model: str,
    run_name: str,
    config: dict,
    model_params: dict,
    measure: str = "cases",
    database_url: Optional[str] = None,
) -> str:
    """
    Read-only preview of the run_name save_walkforward_run() would use for this
    exact config/run_name combination — lets callers (scripts/run_walkforward.py)
    name local files consistently with what will end up in results_unsa_ira,
    including the auto-suffix applied when run_name collides with a different
    configuration.

    Best-effort: unlike save_walkforward_run(), this doesn't take the advisory
    lock, so a concurrent writer could still change the outcome before the real
    insert (which re-resolves under a lock). Fine for local file naming — the
    goal is to avoid clobbering a differently configured run under the same
    name, not to guarantee global uniqueness (that's results_unsa_ira's job).
    """
    engine = get_db_engine(database_url)
    model_params_json = json.dumps(model_params, default=str)
    with engine.connect() as conn:
        _, resolved_name = _resolve_run(
            conn, department, age_group, measure, model, run_name, config, model_params_json
        )
    return resolved_name


def find_existing_run(
    department: str,
    age_group: str,
    model: str,
    config: dict,
    model_params: dict,
    measure: str = "cases",
    database_url: Optional[str] = None,
) -> Optional[int]:
    """
    Return the run_id of an existing run with this exact configuration
    (department, age_group, measure, model, horizon, step, window_type,
    train_size, refit_every, model_params), or None if no such run exists.

    Used both by save_walkforward_run() (to avoid inserting a duplicate) and
    by scripts/run_walkforward.py (to skip retraining entirely when the same
    configuration was already run).
    """
    engine = get_db_engine(database_url)
    with engine.connect() as conn:
        return conn.execute(
            text(_MATCH_CONFIG_QUERY),
            {
                "department": department,
                "age_group": age_group,
                "measure": measure,
                "model": model,
                "horizon": config["horizon"],
                "step": config["step"],
                "window_type": config["window_type"],
                "train_size": config["train_size"],
                "refit_every": config["refit_every"],
                "model_params": json.dumps(model_params, default=str),
            },
        ).scalar()


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
    Insert one walkforward_runs row plus its walkforward_predictions rows —
    unless an identical configuration already exists, in which case that
    existing run_id is reused and nothing new is inserted.

    Two safeguards against duplicate/colliding data, both scoped by an
    advisory lock on (department, age_group, measure, run_name) so concurrent
    callers (e.g. parallel grid-search workers) serialize correctly:

      1. If a run with the exact same configuration (department, age_group,
         measure, model, horizon, step, window_type, train_size, refit_every,
         model_params) already exists under ANY run_name, that run_id is
         reused — no new row, no new predictions.
      2. Otherwise, if `run_name` is already used by a DIFFERENT configuration
         in this (department, age_group, measure) scope — regardless of
         model — the name is auto-suffixed: 'run_name(1)', 'run_name(2)', ...
         This keeps run_name unique across all models within a department/
         age_group, which the rest of the codebase (dict keys, groupby)
         assumes. See db/migrations/0002_walkforward_runs_unique, which backs
         this with a UNIQUE constraint as a defensive fallback.

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
        (run_id, resolved_run_name) — run_id new, or an existing one if reused;
        resolved_run_name equal to run_name unless it collided with a
        different configuration, in which case it's the auto-suffixed name.
    """
    engine = get_db_engine(database_url)
    model_params_json = json.dumps(model_params, default=str)

    with engine.begin() as conn:
        lock_key = f"{department}:{age_group}:{measure}:{run_name}"
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": lock_key})

        existing_run_id, resolved_name = _resolve_run(
            conn, department, age_group, measure, model, run_name, config, model_params_json
        )
        if existing_run_id is not None:
            logger.info(
                f"Identical configuration already exists as run_id={existing_run_id} "
                f"({department}/{age_group}/{measure}/{model}) — reusing it, no new insert."
            )
            return existing_run_id, resolved_name

        run_id = conn.execute(
            text("""
                INSERT INTO results_unsa_ira.walkforward_runs
                    (department, age_group, measure, model, run_name,
                     horizon, step, window_type, train_size, refit_every,
                     model_params, n_steps)
                VALUES
                    (:department, :age_group, :measure, :model, :run_name,
                     :horizon, :step, :window_type, :train_size, :refit_every,
                     CAST(:model_params AS jsonb), :n_steps)
                RETURNING run_id
            """),
            {
                "department": department,
                "age_group": age_group,
                "measure": measure,
                "model": model,
                "run_name": resolved_name,
                "horizon": config["horizon"],
                "step": config["step"],
                "window_type": config["window_type"],
                "train_size": config["train_size"],
                "refit_every": config["refit_every"],
                "model_params": model_params_json,
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
    return run_id, resolved_name
