-- 0002_walkforward_runs_unique / down
ALTER TABLE results_unsa_ira.walkforward_runs
    DROP CONSTRAINT IF EXISTS walkforward_runs_config_unique;
