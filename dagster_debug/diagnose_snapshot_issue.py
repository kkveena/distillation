"""
Diagnostic script for DagsterSnapshotDoesNotExist error.

Error: dagster._core.errors.DagsterSnapshotDoesNotExist:
    Snapshot a95a6528a24c5f0539e6f84e8dedbbc458de7d2d does not exist.

This script helps identify and fix the root cause of missing snapshots
in Dagster's run storage.

Usage:
    python diagnose_snapshot_issue.py

Prerequisites:
    - pip install dagster dagster-postgres (or dagster with your storage backend)
    - Set DAGSTER_HOME environment variable or provide dagster.yaml path
"""

import os
import sys
import json
from datetime import datetime


def get_instance():
    """Get the DagsterInstance from environment."""
    try:
        from dagster import DagsterInstance
        instance = DagsterInstance.get()
        print(f"[OK] Connected to Dagster instance at: {instance.root_directory}")
        return instance
    except Exception as e:
        print(f"[ERROR] Could not connect to Dagster instance: {e}")
        print("  Make sure DAGSTER_HOME is set or dagster.yaml is in the current directory.")
        sys.exit(1)


def check_storage_backend(instance):
    """Identify the storage backend being used."""
    storage = instance.run_storage
    storage_type = type(storage).__name__
    print(f"\n--- Storage Backend ---")
    print(f"  Run storage type: {storage_type}")

    if hasattr(instance, '_event_storage'):
        event_storage_type = type(instance._event_storage).__name__
        print(f"  Event log storage type: {event_storage_type}")

    return storage_type


def find_problematic_runs(instance, target_snapshot_id=None):
    """Find runs that reference missing snapshots."""
    from dagster._core.storage.pipeline_run import RunsFilter, DagsterRunStatus

    print(f"\n--- Checking for Runs with Missing Snapshots ---")

    all_runs = instance.get_runs()
    problematic_runs = []
    snapshot_ids_seen = set()

    for run in all_runs:
        snapshot_id = run.pipeline_snapshot_id
        if snapshot_id:
            snapshot_ids_seen.add(snapshot_id)

            if target_snapshot_id and snapshot_id != target_snapshot_id:
                continue

            try:
                # Try to fetch the snapshot
                instance.get_pipeline_snapshot(snapshot_id)
            except Exception:
                problematic_runs.append({
                    "run_id": run.run_id,
                    "job_name": run.pipeline_name,
                    "status": str(run.status),
                    "snapshot_id": snapshot_id,
                    "created": str(datetime.fromtimestamp(run.create_timestamp))
                        if run.create_timestamp else "unknown",
                })

    if problematic_runs:
        print(f"  [WARNING] Found {len(problematic_runs)} run(s) with missing snapshots:")
        for r in problematic_runs:
            print(f"    Run: {r['run_id'][:12]}... | Job: {r['job_name']} | "
                  f"Status: {r['status']} | Snapshot: {r['snapshot_id'][:12]}... | "
                  f"Created: {r['created']}")
    else:
        print(f"  [OK] All {len(all_runs)} runs have valid snapshots.")

    print(f"\n  Total unique snapshot IDs referenced: {len(snapshot_ids_seen)}")

    return problematic_runs


def check_dagster_version(instance):
    """Check Dagster version and migration status."""
    import dagster

    print(f"\n--- Dagster Version Info ---")
    print(f"  Dagster version: {dagster.__version__}")

    # Check if migrations are needed
    try:
        from dagster._core.storage.runs import SqlRunStorage
        storage = instance.run_storage
        if isinstance(storage, SqlRunStorage):
            print(f"  Storage is SQL-based, checking migration status...")
            # The instance will raise if migrations are needed
            try:
                instance.get_runs(limit=1)
                print(f"  [OK] Run storage appears to be up to date.")
            except Exception as e:
                if "migration" in str(e).lower() or "alembic" in str(e).lower():
                    print(f"  [WARNING] Database migration may be needed: {e}")
                else:
                    raise
    except ImportError:
        pass


def check_for_reexecution_issues(instance):
    """Check for runs that were created via re-execution."""
    print(f"\n--- Re-execution Run Analysis ---")

    all_runs = instance.get_runs()
    reexecution_runs = []

    for run in all_runs:
        tags = run.tags or {}
        parent_run_id = tags.get("dagster/parent_run_id")
        root_run_id = tags.get("dagster/root_run_id")

        if parent_run_id or root_run_id:
            reexecution_runs.append({
                "run_id": run.run_id,
                "parent_run_id": parent_run_id,
                "root_run_id": root_run_id,
                "job_name": run.pipeline_name,
                "status": str(run.status),
            })

    if reexecution_runs:
        print(f"  Found {len(reexecution_runs)} re-execution run(s):")
        for r in reexecution_runs:
            # Check if parent run still exists
            parent_exists = "N/A"
            if r["parent_run_id"]:
                try:
                    parent = instance.get_run_by_id(r["parent_run_id"])
                    parent_exists = "YES" if parent else "NO"
                except Exception:
                    parent_exists = "ERROR"

            print(f"    Run: {r['run_id'][:12]}... | Parent: {r['parent_run_id'][:12] if r['parent_run_id'] else 'None'}... "
                  f"| Parent exists: {parent_exists} | Job: {r['job_name']}")
    else:
        print(f"  No re-execution runs found.")


def suggest_fixes(problematic_runs, storage_type):
    """Suggest fixes based on the diagnosis."""
    print(f"\n{'='*60}")
    print(f"  RECOMMENDED FIXES")
    print(f"{'='*60}")

    if not problematic_runs:
        print("""
  The snapshot issue may be intermittent or environment-specific.
  Common causes:

  1. STALE DAGSTER WEBSERVER/DAEMON CACHE
     - Restart the Dagster webserver (dagit/dagster-webserver)
     - Restart the Dagster daemon
     - Commands:
         dagster-webserver -h 0.0.0.0 -p 3000  (restart webserver)
         dagster-daemon run                      (restart daemon)

  2. CODE LOCATION OUT OF SYNC
     - If using gRPC code locations, reload the code location
     - In Dagster UI: Deployment > Code Locations > Reload
     - Or restart the code server process

  3. CONCURRENT DEPLOYMENT
     - If a new deployment happened while a run was in progress,
       the old snapshot may have been replaced
""")
        return

    print(f"""
  Found {len(problematic_runs)} run(s) referencing missing snapshots.

  FIX 1: WIPE ORPHANED RUNS (if runs are old/not needed)
  -------------------------------------------------------
  Delete the runs that reference missing snapshots.
  This is safe if these runs are already in a terminal state
  (SUCCESS, FAILURE, CANCELED).

  Run this in a Python shell:

    from dagster import DagsterInstance
    instance = DagsterInstance.get()
""")
    for r in problematic_runs:
        print(f"    instance.delete_run('{r['run_id']}')")

    print(f"""
  FIX 2: RE-REGISTER THE SNAPSHOT (if you need to keep the runs)
  ---------------------------------------------------------------
  Launch a fresh run of the same job. This will create a new snapshot.
  Note: re-execution from the old parent run will still fail because
  the old snapshot hash won't match the new code. Use a fresh run instead.

  FIX 3: DATABASE MIGRATION (if you recently upgraded Dagster)
  -------------------------------------------------------------
  Run:
    dagster instance migrate

  This ensures all storage tables are up to date.

  FIX 4: INSTEAD OF RE-EXECUTION, USE A FRESH RUN
  -------------------------------------------------
  The error occurs specifically during RE-EXECUTION (launchpad
  "Re-execute" button). Instead of re-executing from the failed run:

  a) Go to the Job page in Dagster UI
  b) Click "Launch Run" (not "Re-execute" from a previous run)
  c) Configure with the same parameters
  d) Launch as a fresh run

  This avoids the snapshot lookup entirely.

  FIX 5: CHECK AND REPAIR DATABASE STORAGE
  -----------------------------------------
  If using PostgreSQL:
    SELECT COUNT(*) FROM snapshots;
    SELECT DISTINCT pipeline_snapshot_id FROM runs
    WHERE pipeline_snapshot_id NOT IN (SELECT snapshot_id FROM snapshots);

  If using SQLite:
    Check if $DAGSTER_HOME/storage/ directory has the run database
    and it hasn't been corrupted or accidentally deleted.
""")


def run_direct_db_check():
    """Attempt direct database checks if possible."""
    print(f"\n--- Direct Database Snapshot Check ---")

    dagster_home = os.environ.get("DAGSTER_HOME", "")
    if not dagster_home:
        print("  DAGSTER_HOME not set. Checking common locations...")
        common_paths = [
            os.path.expanduser("~/.dagster"),
            "/opt/dagster/dagster_home",
            "/var/dagster",
        ]
        for p in common_paths:
            if os.path.isdir(p):
                dagster_home = p
                break

    if dagster_home:
        print(f"  Dagster home: {dagster_home}")

        # Check dagster.yaml for storage config
        yaml_path = os.path.join(dagster_home, "dagster.yaml")
        if os.path.exists(yaml_path):
            print(f"  Found dagster.yaml at: {yaml_path}")
            with open(yaml_path, 'r') as f:
                content = f.read()
            print(f"  Storage config preview:")
            for line in content.split('\n'):
                if any(keyword in line.lower() for keyword in
                       ['storage', 'postgres', 'mysql', 'sqlite', 'run_storage',
                        'event_log', 'schedule']):
                    print(f"    {line}")
        else:
            print(f"  No dagster.yaml found at {yaml_path}")

        # Check for SQLite databases
        storage_dir = os.path.join(dagster_home, "storage")
        if os.path.isdir(storage_dir):
            print(f"\n  SQLite storage directory contents:")
            for f in os.listdir(storage_dir):
                fpath = os.path.join(storage_dir, f)
                size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                print(f"    {f} ({size / 1024:.1f} KB)")
    else:
        print("  Could not locate Dagster home directory.")
        print("  Set DAGSTER_HOME environment variable and re-run.")


def main():
    print("=" * 60)
    print("  Dagster DagsterSnapshotDoesNotExist Diagnostic Tool")
    print("=" * 60)

    # The specific snapshot ID from the error
    TARGET_SNAPSHOT = "a95a6528a24c5f0539e6f84e8dedbbc458de7d2d"
    print(f"\n  Target snapshot: {TARGET_SNAPSHOT}")

    # Try direct DB check first (doesn't require full Dagster)
    run_direct_db_check()

    try:
        instance = get_instance()
        check_dagster_version(instance)
        storage_type = check_storage_backend(instance)
        problematic_runs = find_problematic_runs(instance, TARGET_SNAPSHOT)
        check_for_reexecution_issues(instance)
        suggest_fixes(problematic_runs, storage_type)
    except ImportError as e:
        print(f"\n[WARNING] Dagster not installed in this environment: {e}")
        print("  Install dagster to run full diagnostics:")
        print("    pip install dagster dagster-postgres")
        print("\n  Providing general fix recommendations instead...")
        suggest_fixes([], "unknown")
    except Exception as e:
        print(f"\n[ERROR] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Providing general fix recommendations instead...")
        suggest_fixes([], "unknown")


if __name__ == "__main__":
    main()
