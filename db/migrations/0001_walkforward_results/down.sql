-- 0001_walkforward_results / down
-- Reverses up.sql. Drops the whole 'results_unsa_ira' schema (and everything
-- in it) — safe as long as nothing else has been added to that schema outside
-- this migration.

DROP SCHEMA IF EXISTS results_unsa_ira CASCADE;
