#!/bin/bash
# Designed to be SOURCED
# use
# source ./src/run.sh 1   # CPU 16
# source ./src/run.sh 2   # CPUs 16-17
# source ./src/run.sh 4   # CPUs 16-19
# source ./src/run.sh 8   # CPUs 16-23
# source ./src/run.sh 16  # CPUs 16-31

NCORES="${1:-16}"

START_CPU=16
END_CPU=$((START_CPU + NCORES - 1))

CPUSET="${START_CPU}-${END_CPU}"

CG=/sys/fs/cgroup/amir_bench

# Don't allow more than the 16 CPUs you reserved.
if (( NCORES < 1 || NCORES > 16 )); then
    echo "Error: number of CPUs must be between 1 and 16"
    return 1
fi

# --- partition must exist and be root ---
[[ "$(cat "$CG/cpuset.cpus.partition" 2>/dev/null)" == "root" ]] || {
    echo "partition missing/degraded — rerun setup"
    return 1
}

# --- enroll THIS shell ---
MYPID=$$

sudo sh -c "echo $MYPID > $CG/cgroup.procs" || {
    echo "enrollment write failed"
    return 1
}

# --- verify ---
grep -qx "$MYPID" "$CG/cgroup.procs" || {
    echo "PID $MYPID not in cgroup"
    return 1
}

allowed=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')

[[ "$allowed" == "16-31" ]] || {
    echo "mask is $allowed, expected 16-31"
    return 1
}

echo "Reserved CPUs: $allowed"
echo "Running on $NCORES CPU(s): $CPUSET"

export OMP_NUM_THREADS="$NCORES"
export OPENBLAS_NUM_THREADS="$NCORES"
export MKL_NUM_THREADS="$NCORES"

# evict the OS level page cache
# sync
# echo 3 | sudo tee /proc/sys/vm/drop_caches
#
vmtouch -v /upb/users/a/amirk/scratch/UPB_phd/RAG/RAGPerf/RAGPerf_dbs/lance_pdfimage/vidore_v3_physics_ivfsq.lance
# vmtouch -v /upb/users/a/amirk/scratch/UPB_phd/RAG/RAGPerf/RAGPerf_dbs/lance_pdfimage

taskset -c "$CPUSET" python ./src/run_new.py \
    --config ./config/pdfimage/lance_query_pdfimage_rerank_Vidore3_physiscs_IVF_SQ.yaml\
    --msys-config ./config/monitor/example_config.yaml