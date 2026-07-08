#!/bin/bash
# NOTE: designed to be SOURCED. No set -e (it would arm your login shell);
# explicit checks with `return` instead.

CORES="${1:-16}"
CG=/sys/fs/cgroup/amir_bench

# --- partition must exist and be root ---
[[ "$(cat $CG/cpuset.cpus.partition 2>/dev/null)" == "root" ]] || {
    echo "partition missing/degraded — rerun setup"; return 1; }

# --- enroll THIS shell, explicit PID ---
MYPID=$$
sudo sh -c "echo $MYPID > $CG/cgroup.procs" || {
    echo "enrollment write failed"; return 1; }

# --- verify by membership, not by inference ---
grep -qx "$MYPID" $CG/cgroup.procs || {
    echo "PID $MYPID not in cgroup after write"; return 1; }
allowed=$(grep Cpus_allowed_list /proc/self/status | cut -f2)
[[ "$allowed" == "16-31" ]] || { echo "mask is $allowed, expected 16-31"; return 1; }
echo "enrolled: shell $MYPID on $allowed"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

taskset -c "$CORES" python ./src/run_new.py \
    --config ./config/pdfimage/lance_query_pdfimage_rerank_Vidore3_physiscs.yaml \
    --msys-config ./config/monitor/example_config.yaml