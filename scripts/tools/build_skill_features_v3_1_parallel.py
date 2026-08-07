"""Parallel walk-forward builder for v3.1 skill features.

Spawns N worker processes (each single-threaded so they don't fight for cores),
each handling a contiguous block of months. After all workers finish, merges
their per-shard parquets into data/processed/skill_features_v3_1.parquet.

Defaults to 8 workers on an M-series Mac with ≥12 performance cores. Tune via
--workers if you have a different machine.

Usage:
    python scripts/build_skill_features_v3_1_parallel.py [--workers N]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)"
    )
    args = parser.parse_args()
    n = args.workers

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".conda" / "bin" / "python")
    module = "ufc_pred.features.skill_v3_1_pipeline"

    # Prevent macOS idle/system sleep while the launcher itself runs.
    # `caffeinate -i -w PID` blocks idle sleep until our PID exits.
    # We spawn caffeinate as our child so it dies with us.
    if sys.platform == "darwin":
        my_pid = os.getpid()
        caffeinate = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(my_pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[launcher] caffeinate PID {caffeinate.pid} guarding against idle sleep")

    # Single-threaded math libs per worker so N workers ≠ N×K threads.
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        }
    )

    procs = []
    t0 = time.time()
    for shard_id in range(n):
        log_file = LOG_DIR / f"v3_1_shard{shard_id}.log"
        cmd = [
            python,
            "-m",
            module,
            "--shard-id",
            str(shard_id),
            "--n-shards",
            str(n),
            "--no-progress",
        ]
        print(f"[launcher] shard {shard_id}/{n}: {shlex.join(cmd)}")
        print(f"[launcher]   log: {log_file}")
        f = open(log_file, "w")
        p = subprocess.Popen(cmd, env=base_env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs.append((shard_id, p, f, log_file))

    print(f"[launcher] {n} workers spawned. Waiting...")

    # Wait for all to finish; report each as it completes.
    failed = []
    for shard_id, p, f, log_file in procs:
        rc = p.wait()
        f.close()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[launcher] shard {shard_id}: {status}  (t={elapsed / 60:.1f} min)")
        if rc != 0:
            failed.append(shard_id)

    if failed:
        print(f"[launcher] {len(failed)} shard(s) failed: {failed}")
        print(f"[launcher] Check logs in {LOG_DIR}. Not merging.")
        sys.exit(1)

    # Merge.
    print("[launcher] Merging shards...")
    rc = subprocess.call(
        [python, "-m", module, "--n-shards", str(n), "--merge"],
        env=base_env,
        cwd=str(ROOT),
    )
    if rc != 0:
        print("[launcher] Merge failed.")
        sys.exit(1)

    total_min = (time.time() - t0) / 60
    print(f"[launcher] DONE. Total wall time: {total_min:.1f} min.")
    print("[launcher] Output: data/processed/skill_features_v3_1.parquet")


if __name__ == "__main__":
    main()
