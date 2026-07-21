-- 0003_walkforward_holdout / down

ALTER TABLE results_unsa_ira.walkforward_runs
    DROP CONSTRAINT walkforward_runs_config_unique;

ALTER TABLE results_unsa_ira.walkforward_runs
    ADD CONSTRAINT walkforward_runs_config_unique
    UNIQUE (department, age_group, measure, model, horizon, step,
            window_type, train_size, refit_every, model_params);

ALTER TABLE results_unsa_ira.walkforward_runs
    DROP COLUMN holdout_start;
