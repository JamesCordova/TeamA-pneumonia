-- 0001_walkforward_results / up
--
-- Two tables in a dedicated schema, kept separate from the existing
-- source-data tables (iras_*, population_*, ...) that live in 'public' on the
-- same database:
--
--   results_unsa_ira.walkforward_runs         one row per WalkForwardValidator
--                                              execution (a grid-search trial
--                                              = one row here)
--   results_unsa_ira.walkforward_predictions  one row per individual forecast
--                                              made during a run (raw
--                                              actual/predicted, never
--                                              aggregated)
--
-- All comparison modes (horizon / macroaverage / microaverage / cumulative,
-- and --year) are derived from walkforward_predictions at query time —
-- nothing pre-aggregated is stored, so nothing can go stale.

CREATE SCHEMA IF NOT EXISTS results_unsa_ira;

CREATE TABLE results_unsa_ira.walkforward_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    department    TEXT NOT NULL,
    age_group     TEXT NOT NULL,
    measure       TEXT NOT NULL DEFAULT 'cases',
    model         TEXT NOT NULL,
    run_name      TEXT NOT NULL,
    horizon       INTEGER NOT NULL,
    step          INTEGER NOT NULL,
    window_type   TEXT NOT NULL,
    train_size    INTEGER NOT NULL,
    refit_every   INTEGER NOT NULL,
    model_params  JSONB NOT NULL DEFAULT '{}'::jsonb,
    n_steps       INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  results_unsa_ira.walkforward_runs IS
    'One row per run_walkforward.py execution. Never overwritten — a rerun '
    '(e.g. a new grid-search trial) always inserts a new row, so history is '
    'never lost the way *_walkforward_metrics.json is today.';
COMMENT ON COLUMN results_unsa_ira.walkforward_runs.measure IS
    'Target variable being forecast: ''cases'' (the only one implemented '
    'today — get_departmental_data() only returns case counts), with '
    '''hospitalizations''/''deaths''/etc. as future options. Lives on the run, '
    'not on walkforward_predictions, because one run always forecasts exactly '
    'one measure — same one-to-many relationship as department/age_group/model.';
COMMENT ON COLUMN results_unsa_ira.walkforward_runs.model_params IS
    'Model-specific hyperparameters (n_estimators, learning_rate, sarima_order, '
    '...) — kept as JSONB rather than per-model columns because the hyperparameter '
    'set varies by model type. Filter with model_params->>''key''.';
COMMENT ON COLUMN results_unsa_ira.walkforward_runs.horizon IS
    'Weeks-ahead forecast on each call to the model. horizon > step is valid '
    'and intentional (overlapping evaluation windows) — see walkforward_predictions.';
COMMENT ON COLUMN results_unsa_ira.walkforward_runs.step IS
    'Weeks the training cutoff advances between evaluations. When step < '
    'horizon, evaluation windows overlap — see walkforward_predictions.';

CREATE INDEX idx_walkforward_runs_lookup
    ON results_unsa_ira.walkforward_runs (department, age_group, measure, model);
CREATE INDEX idx_walkforward_runs_model_params
    ON results_unsa_ira.walkforward_runs USING GIN (model_params);


CREATE TABLE results_unsa_ira.walkforward_predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES results_unsa_ira.walkforward_runs (run_id) ON DELETE CASCADE,
    step_idx        INTEGER NOT NULL,
    horizon_offset  INTEGER NOT NULL,
    date            DATE NOT NULL,
    actual          DOUBLE PRECISION,
    predicted       DOUBLE PRECISION,
    UNIQUE (run_id, step_idx, horizon_offset)
);

COMMENT ON TABLE  results_unsa_ira.walkforward_predictions IS
    'One row per individual forecast (raw, never aggregated). Deliberately NOT '
    'deduplicated by date: when horizon > step, the same calendar date is '
    'forecast by multiple steps at different horizon_offsets, and both rows '
    'are kept. Metrics (mae, rmse, r2, ...) are computed on demand by grouping '
    'these rows however a given comparison mode needs (by horizon_offset, by '
    'step_idx, pooled, or ordered by date) — never stored pre-computed.';
COMMENT ON COLUMN results_unsa_ira.walkforward_predictions.actual IS
    'Snapshotted at evaluation time, not looked up live from the source series. '
    'Epidemiological case counts get revised after the fact; freezing the value '
    'that existed when this run was evaluated keeps the backtest reproducible '
    'even if the source data changes later.';
COMMENT ON COLUMN results_unsa_ira.walkforward_predictions.step_idx IS
    'Which walk-forward iteration produced this row — needed to compute '
    'per-step metrics for macroaverage (mean across steps, excluding '
    'non-additive ratio metrics like r2).';
COMMENT ON COLUMN results_unsa_ira.walkforward_predictions.horizon_offset IS
    'Weeks-ahead lead time of this specific prediction (1..horizon) — needed '
    'to group by horizon for horizon mode.';

CREATE INDEX idx_walkforward_predictions_run_horizon
    ON results_unsa_ira.walkforward_predictions (run_id, horizon_offset);
CREATE INDEX idx_walkforward_predictions_run_date
    ON results_unsa_ira.walkforward_predictions (run_id, date);
