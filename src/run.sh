#!/bin/bash
set -euo pipefail

# Usage:
#   ./run.sh 16
#   ./run.sh 16-31
#   ./run.sh 16,18,20-23

CORES="${1:-16}"

# --- enroll this shell in the benchmark partition ---
sudo sh -c "echo $$ > /sys/fs/cgroup/amir_bench/cgroup.procs"

# --- verify, don't trust ---
allowed=$(grep Cpus_allowed_list /proc/self/status | cut -f2)
[[ "$allowed" == "16-31" ]] || { echo "NOT ENROLLED (mask=$allowed)"; exit 1; }

# --- one owner of parallelism: the executor ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

taskset -c "$CORES" python ./src/run_new.py \
    --config ./config/pdfimage/lance_query_pdfimage_rerank_Vidore3_physiscs.yaml \
    --msys-config ./config/monitor/example_config.yaml