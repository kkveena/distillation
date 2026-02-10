-- ============================================================
-- Dagster DagsterSnapshotDoesNotExist - Database Repair Script
-- ============================================================
-- Target snapshot: a95a6528a24c5f0539e6f84e8dedbbc458de7d2d
--
-- Run these queries against your Dagster PostgreSQL database.
-- ALWAYS back up before making changes.
-- ============================================================

-- Step 1: Backup first!
-- pg_dump -h HOST -U USER -d dagster_db > dagster_backup_$(date +%Y%m%d).sql

-- ============================================================
-- DIAGNOSTIC QUERIES
-- ============================================================

-- 1. Check if the target snapshot exists
SELECT snapshot_id, snapshot_type, length(snapshot_body) as body_size
FROM snapshots
WHERE snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d';

-- 2. Count total snapshots
SELECT snapshot_type, COUNT(*) as count
FROM snapshots
GROUP BY snapshot_type;

-- 3. Find ALL runs referencing the missing snapshot
SELECT run_id, pipeline_name, status, create_timestamp, update_timestamp,
       pipeline_snapshot_id
FROM runs
WHERE pipeline_snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d';

-- 4. Find ALL orphaned runs (runs referencing non-existent snapshots)
SELECT r.run_id, r.pipeline_name, r.status, r.create_timestamp,
       r.pipeline_snapshot_id
FROM runs r
LEFT JOIN snapshots s ON r.pipeline_snapshot_id = s.snapshot_id
WHERE r.pipeline_snapshot_id IS NOT NULL
  AND s.snapshot_id IS NULL
ORDER BY r.create_timestamp DESC;

-- 5. Count orphaned vs total runs
SELECT
  COUNT(*) as total_runs,
  SUM(CASE WHEN s.snapshot_id IS NULL AND r.pipeline_snapshot_id IS NOT NULL
      THEN 1 ELSE 0 END) as orphaned_runs
FROM runs r
LEFT JOIN snapshots s ON r.pipeline_snapshot_id = s.snapshot_id;

-- 6. Check re-execution lineage for the affected runs
SELECT r.run_id, r.pipeline_name, r.status,
       rt_parent.value as parent_run_id,
       rt_root.value as root_run_id
FROM runs r
LEFT JOIN run_tags rt_parent ON r.run_id = rt_parent.run_id
  AND rt_parent.key = 'dagster/parent_run_id'
LEFT JOIN run_tags rt_root ON r.run_id = rt_root.run_id
  AND rt_root.key = 'dagster/root_run_id'
WHERE r.pipeline_snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d';


-- ============================================================
-- REPAIR QUERIES (choose one approach)
-- ============================================================

-- APPROACH A: Delete orphaned runs that reference missing snapshots
-- Use this if the affected runs are old and you don't need them.
-- Uncomment to execute:

-- BEGIN;
--
-- -- Delete associated run tags first (foreign key constraint)
-- DELETE FROM run_tags
-- WHERE run_id IN (
--   SELECT r.run_id FROM runs r
--   LEFT JOIN snapshots s ON r.pipeline_snapshot_id = s.snapshot_id
--   WHERE r.pipeline_snapshot_id IS NOT NULL
--     AND s.snapshot_id IS NULL
--     AND r.status IN ('SUCCESS', 'FAILURE', 'CANCELED')
-- );
--
-- -- Delete associated event logs
-- DELETE FROM event_logs
-- WHERE run_id IN (
--   SELECT r.run_id FROM runs r
--   LEFT JOIN snapshots s ON r.pipeline_snapshot_id = s.snapshot_id
--   WHERE r.pipeline_snapshot_id IS NOT NULL
--     AND s.snapshot_id IS NULL
--     AND r.status IN ('SUCCESS', 'FAILURE', 'CANCELED')
-- );
--
-- -- Delete the orphaned runs
-- DELETE FROM runs
-- WHERE pipeline_snapshot_id IS NOT NULL
--   AND pipeline_snapshot_id NOT IN (SELECT snapshot_id FROM snapshots)
--   AND status IN ('SUCCESS', 'FAILURE', 'CANCELED');
--
-- COMMIT;


-- APPROACH B: Null out snapshot references (keeps runs, prevents re-execution)
-- Use this if you want to keep the run history but don't need re-execution.
-- Uncomment to execute:

-- BEGIN;
--
-- UPDATE runs
-- SET pipeline_snapshot_id = NULL
-- WHERE pipeline_snapshot_id NOT IN (SELECT snapshot_id FROM snapshots)
--   AND pipeline_snapshot_id IS NOT NULL;
--
-- COMMIT;


-- APPROACH C: Fix only the specific snapshot causing the error
-- Uncomment to execute:

-- BEGIN;
--
-- UPDATE runs
-- SET pipeline_snapshot_id = NULL
-- WHERE pipeline_snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d';
--
-- COMMIT;


-- ============================================================
-- VERIFICATION (run after applying a fix)
-- ============================================================

-- Verify no orphaned runs remain
SELECT COUNT(*) as remaining_orphaned_runs
FROM runs r
LEFT JOIN snapshots s ON r.pipeline_snapshot_id = s.snapshot_id
WHERE r.pipeline_snapshot_id IS NOT NULL
  AND s.snapshot_id IS NULL;
