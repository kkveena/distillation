# Dagster `DagsterSnapshotDoesNotExist` - Troubleshooting Guide

## Error

```
dagster._core.errors.DagsterSnapshotDoesNotExist:
  Snapshot a95a6528a24c5f0539e6f84e8dedbbc458de7d2d does not exist
```

### Stack Trace (from the reported issue)

```
sql_run_storage.py:134, in add_run
  → raise DagsterSnapshotDoesNotExist

instance/__init__.py:1600, in create_run
  → dagster_run = self._run_storage.add_run(dagster_run)

instance/__init__.py:1681, in create_reexecuted_run
  → return self.create_run(...)

launch_execution.py, in launch_reexecution_from_parent_run
  → run = instance.create_reexecuted_run(...)

mutation.py:430, in mutate
  → return launch_reexecution_from_parent_run(...)
```

---

## Root Cause

The error happens during **re-execution** of a Dagster job. The flow is:

1. User clicks "Re-execute" on a previous run in the Dagster UI
2. Dagster looks up the parent run's **pipeline/job snapshot** (a hash of the serialized job definition)
3. The snapshot ID `a95a6528...` is referenced in the `runs` table but **does not exist** in the `snapshots` table
4. Dagster raises `DagsterSnapshotDoesNotExist`

A snapshot is a frozen representation of the job definition at the time a run was created. It enables Dagster to show the exact code structure that was used for any historical run.

---

## Likely Causes

### 1. Database Was Partially Wiped or Reset
If the snapshots table was truncated or the database was restored from a partial backup, run records may reference snapshot IDs that no longer exist.

### 2. Dagster Version Upgrade Without Migration
After upgrading Dagster, the schema may have changed. If `dagster instance migrate` was not run, the snapshots table may be in an inconsistent state.

### 3. Job Code Changed and Old Snapshot Was Not Preserved
When the job definition code changes, a new snapshot is created. If the old snapshot was cleaned up (or never properly stored due to an error), re-executing from the old run will fail.

### 4. Multiple Dagster Instances or Deployments Sharing Storage
If two Dagster deployments (e.g., staging and production, or two code servers) share the same database but have different code, snapshots may be inconsistent.

### 5. Storage Backend Issues
- SQLite: file corruption, disk space issues, or accidental deletion of storage files
- PostgreSQL: connection issues during snapshot write, or manual table operations

---

## Fixes

### Fix 1: Launch a Fresh Run Instead of Re-executing (Immediate Workaround)

The error only occurs during **re-execution**. To work around it:

1. Go to the **Job** page in Dagster UI (not the failed run page)
2. Click **"Launchpad"** or **"Launch Run"**
3. Configure with the same parameters as the failed run
4. Launch as a **new run**

This bypasses the snapshot lookup entirely.

### Fix 2: Run Database Migration

```bash
export DAGSTER_HOME=/path/to/dagster/home
dagster instance migrate
```

This ensures all storage tables match the current Dagster version's expected schema.

### Fix 3: Delete Orphaned Runs

If the problematic runs are old and no longer needed:

```python
from dagster import DagsterInstance

instance = DagsterInstance.get()

# Find the run that references the missing snapshot
runs = instance.get_runs()
for run in runs:
    if run.pipeline_snapshot_id == "a95a6528a24c5f0539e6f84e8dedbbc458de7d2d":
        print(f"Deleting run {run.run_id} (job: {run.pipeline_name}, status: {run.status})")
        instance.delete_run(run.run_id)
```

### Fix 4: Repair the Snapshots Table Directly (PostgreSQL)

See `repair_snapshots.sql` for database-level queries.

**Find orphaned runs:**
```sql
SELECT r.run_id, r.pipeline_name, r.status, r.pipeline_snapshot_id
FROM runs r
WHERE r.pipeline_snapshot_id IS NOT NULL
  AND r.pipeline_snapshot_id NOT IN (
    SELECT snapshot_id FROM snapshots
  );
```

**Option A - Delete orphaned runs:**
```sql
DELETE FROM runs
WHERE pipeline_snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d'
  AND status IN ('SUCCESS', 'FAILURE', 'CANCELED');
```

**Option B - Null out the snapshot reference (keeps runs but breaks re-execution for them):**
```sql
UPDATE runs
SET pipeline_snapshot_id = NULL
WHERE pipeline_snapshot_id = 'a95a6528a24c5f0539e6f84e8dedbbc458de7d2d';
```

### Fix 5: Reload Code Location

If the issue is caused by a stale code location:

1. In Dagster UI: **Deployment** > **Code Locations**
2. Click **Reload** on the relevant code location
3. Or restart the code server process:
   ```bash
   # If using gRPC code server
   dagster api grpc -h 0.0.0.0 -p 4000 -m your_module

   # If using dagster-webserver directly
   dagster-webserver -h 0.0.0.0 -p 3000
   ```

### Fix 6: Restart All Dagster Services

```bash
# Stop all services
# (method depends on your deployment - systemd, docker-compose, k8s, etc.)

# Restart webserver
dagster-webserver -h 0.0.0.0 -p 3000 &

# Restart daemon
dagster-daemon run &
```

---

## Prevention

1. **Always run `dagster instance migrate`** after upgrading Dagster
2. **Use PostgreSQL** in production instead of SQLite for more reliable storage
3. **Back up your database** before major version upgrades
4. **Avoid sharing run storage** between different Dagster deployments
5. **Use `dagster instance info`** to verify your storage configuration:
   ```bash
   dagster instance info
   ```

---

## AWS MFA Authentication Issue (Also Mentioned in the Email)

The email also mentions AWS MFA authentication token issues. If your Dagster jobs interact with AWS:

1. Ensure your AWS credentials are not expired:
   ```bash
   aws sts get-caller-identity
   ```

2. If using MFA, refresh the session token:
   ```bash
   aws sts get-session-token \
     --serial-number arn:aws:iam::ACCOUNT_ID:mfa/USERNAME \
     --token-code TOKEN_FROM_MFA_DEVICE
   ```

3. Update the Dagster resource configuration with fresh credentials, or use IAM roles instead of long-lived credentials with MFA.

---

## Diagnostic Script

Run the included diagnostic script:

```bash
export DAGSTER_HOME=/path/to/dagster/home
python diagnose_snapshot_issue.py
```

This will:
- Check your Dagster version and storage configuration
- Find runs with missing snapshots
- Analyze re-execution dependencies
- Suggest specific fixes for your environment
