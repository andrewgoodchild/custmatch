-- Least-privilege roles for a custmatch-server database (Postgres). Run once as the owner, after
-- the first `custmatch-server` start has created the tables; change the passwords.
--   custmatch_api     the API and dashboard: read the results, write real-time matches and the lease
--   custmatch_worker  the batch worker and loader: load source records, publish runs
-- The owner (who runs the migrations) keeps DDL; neither role can alter the schema.

CREATE ROLE custmatch_api LOGIN PASSWORD 'change-me';
CREATE ROLE custmatch_worker LOGIN PASSWORD 'change-me';

GRANT USAGE ON SCHEMA public TO custmatch_api, custmatch_worker;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO custmatch_api, custmatch_worker;
REVOKE SELECT ON api_keys FROM custmatch_worker;                 -- the worker never checks keys

-- the API: real-time records go in as source records of today's API load, with their decision;
-- the matcher's merges; stewards' decisions (a newer decision on a pair replaces the old one)
GRANT INSERT, UPDATE ON loads, source_records, source_record_versions, realtime_records, realtime_merges, leases TO custmatch_api;
GRANT INSERT, UPDATE, DELETE ON overrides TO custmatch_api;
-- the steward tools: claims on review pairs, pinned golden values, splits, and the audit log
GRANT INSERT, UPDATE, DELETE ON review_claims, golden_pins TO custmatch_api;
GRANT INSERT ON realtime_moves, steward_log TO custmatch_api;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO custmatch_api;

-- the worker: loads, runs and their results (it creates and drops a run's partitions, so it owns them)
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO custmatch_worker;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO custmatch_worker;
GRANT CREATE ON SCHEMA public TO custmatch_worker;

-- personal data: neither role should be used for ad-hoc analysis; give analysts a role that reads
-- only customers (golden records) or nothing, as your policy requires
