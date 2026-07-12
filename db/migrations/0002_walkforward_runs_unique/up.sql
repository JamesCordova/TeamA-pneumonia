-- 0002_walkforward_runs_unique / up
--
-- A run is uniquely identified by WHAT it did (its full configuration), not
-- by the run_name label a user happened to type. Two rows with the exact
-- same department/age_group/measure/model/horizon/step/window_type/
-- train_size/refit_every/model_params ARE the same run.
--
-- run_name is deliberately EXCLUDED from this constraint — uniqueness must
-- not depend on what the run was called. pneumonia/evaluation/results_db.py's
-- save_walkforward_run() checks for an identical configuration before
-- inserting and reuses the existing run_id if found, so this constraint acts
-- as a defensive backstop (belt-and-suspenders) rather than the primary
-- dedup mechanism.

ALTER TABLE results_unsa_ira.walkforward_runs
    ADD CONSTRAINT walkforward_runs_config_unique
    UNIQUE (department, age_group, measure, model, horizon, step,
            window_type, train_size, refit_every, model_params);
