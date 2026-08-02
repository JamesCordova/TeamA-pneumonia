-- 0003_walkforward_holdout / up
--
-- Adds holdout_start: the calendar date (inclusive) marking where a
-- deliberately protected "holdout" evaluation window begins for this run's
-- configuration. NULL for every run created before this migration, and for
-- any future run that intentionally opts out of holdout protection — those
-- are simply a different configuration from one that has holdout_start
-- set, never conflated with it.
--
-- Why this needs to be part of identity (not just informational): a run
-- with holdout_start='2022-01-01' and one with holdout_start=NULL, sharing
-- every other column, cover DIFFERENT sets of evaluated dates. Dates before
-- holdout_start are the "search" range a model can be selected against;
-- dates >= holdout_start are only ever scored once, after selection, never
-- used to choose between configurations (see
-- pneumonia.pipelines.walkforward_runner, which incrementally extends a
-- single run's walkforward_predictions coverage as pre-holdout / holdout
-- ranges are requested, in either order). Without holdout_start in the
-- uniqueness check, find_existing_run() could return a run whose stored
-- predictions cover the wrong span for what was actually asked.

ALTER TABLE results_unsa_ira.walkforward_runs
    ADD COLUMN holdout_start DATE NULL;

COMMENT ON COLUMN results_unsa_ira.walkforward_runs.holdout_start IS
    'First date (inclusive) reserved as a protected holdout evaluation '
    'window for this run''s configuration, or NULL if this run has no '
    'holdout protection (every run created before this column existed, '
    'plus any future run that opts out). Dates before holdout_start are '
    'the "search" range models can be selected against; dates >= '
    'holdout_start are only ever scored once, after selection, never used '
    'to choose between configurations. See pneumonia.pipelines.walkforward_runner.';

-- Drop and recreate the composite uniqueness constraint from
-- 0002_walkforward_runs_unique to include holdout_start — two rows
-- differing only in holdout_start are genuinely different runs (different
-- evaluated date coverage), not duplicates of each other. NULLS NOT
-- DISTINCT (PostgreSQL 15+) so two holdout_start=NULL rows still can't
-- silently duplicate the same configuration — NULL is treated as its own
-- comparable value here, not as "unknown, never equal to anything".
ALTER TABLE results_unsa_ira.walkforward_runs
    DROP CONSTRAINT walkforward_runs_config_unique;

ALTER TABLE results_unsa_ira.walkforward_runs
    ADD CONSTRAINT walkforward_runs_config_unique
    UNIQUE NULLS NOT DISTINCT
    (department, age_group, measure, model, horizon, step,
     window_type, train_size, refit_every, model_params, holdout_start);
