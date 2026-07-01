#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_scaling_experiments.sh
# Sweeps max_rerank_worker over THREAD_COUNTS, runs the RAGPerf pipeline for
# each setting, and renames the output folder to physics_second_<N>T_gpuembed.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
THREAD_COUNTS=(1 2 4 8 16 32 64 128)

RAGPERF_DIR="/local/amirk/RAGPerf"
CONFIG_FILE="${RAGPERF_DIR}/config/pdfimage/lance_query_pdfimage_rerank_Vidore3_physiscs.yaml"
MONITOR_CONFIG="${RAGPERF_DIR}/config/monitor/example_config.yaml"
OUTPUT_DIR="${RAGPERF_DIR}/src/output"
RUN_SCRIPT="${RAGPERF_DIR}/src/run_new.py"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Replace the max_rerank_worker value in the config file in-place.
# Uses sed so the surrounding indentation is preserved exactly.
set_num_threads() {
    local n=$1
    # matches any amount of leading whitespace, the key, optional spaces, colon,
    # optional spaces, then any integer value
    sed -i -E "s/^(\s*max_rerank_worker\s*:\s*)[0-9]+/\1${n}/" "$CONFIG_FILE"
}

# After the run, find the output folder whose name (ISO-8601 timestamp) is
# closest to the given epoch second.
find_output_folder() {
    local ref_epoch=$1          # epoch seconds at which the run started
    local best_name=""
    local best_diff=999999999

    for folder in "$OUTPUT_DIR"/*/; do
        folder="${folder%/}"
        name=$(basename "$folder")
        # parse timestamp like 2026-06-30T22:01:52+0200
        # strip timezone offset, replace T with space, parse with date
        ts_clean=$(echo "$name" | sed 's/+[0-9]\{4\}$//' | tr 'T' ' ')
        folder_epoch=$(date -d "$ts_clean" +%s 2>/dev/null || echo "")
        if [[ -z "$folder_epoch" ]]; then
            continue   # skip folders that don't parse as timestamps
        fi
        diff=$(( folder_epoch - ref_epoch ))
        diff=${diff#-}   # abs value
        if (( diff < best_diff )); then
            best_diff=$diff
            best_name=$(basename "$folder")
        fi
    done

    echo "$best_name"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

log "Starting scaling sweep: threads = ${THREAD_COUNTS[*]}"
log "Config  : $CONFIG_FILE"
log "Output  : $OUTPUT_DIR"
echo ""

for N in "${THREAD_COUNTS[@]}"; do

    TARGET_NAME="physics_second_${N}T_gpuembed"
    TARGET_PATH="${OUTPUT_DIR}/${TARGET_NAME}"

    # skip if already done
    if [[ -d "$TARGET_PATH" ]]; then
        log "SKIP  ${N}T — '${TARGET_NAME}' already exists"
        continue
    fi

    # ── 1. patch config ───────────────────────────────────────────────────────
    log "Setting max_rerank_worker = ${N} in config..."
    set_num_threads "$N"

    # verify the change landed correctly
    actual=$(grep -E '^\s*max_rerank_worker\s*:' "$CONFIG_FILE" | \
             sed -E 's/.*:\s*//')
    if [[ "$actual" != "$N" ]]; then
        log "ERROR: config patch failed — got '${actual}', expected '${N}'"
        exit 1
    fi
    log "Config verified: max_rerank_worker = ${actual}"

    # ── 2. record start time and run ──────────────────────────────────────────
    START_EPOCH=$(date +%s)
    log "Starting ${N}T run at $(date '+%Y-%m-%d %H:%M:%S')..."
    echo "────────────────────────────────────────────────────────────────"

    cd "$RAGPERF_DIR"
    time python "${RUN_SCRIPT}" \
        --config  "${CONFIG_FILE}" \
        --msys-config "${MONITOR_CONFIG}" \
        2>&1 | tee "${OUTPUT_DIR}/run_${N}T.log"

    RUN_EXIT=${PIPESTATUS[0]}
    echo "────────────────────────────────────────────────────────────────"

    if [[ $RUN_EXIT -ne 0 ]]; then
        log "ERROR: run exited with code ${RUN_EXIT} for ${N}T — aborting sweep"
        exit $RUN_EXIT
    fi

    # ── 3. find and rename the output folder ─────────────────────────────────
    log "Locating output folder (started at epoch ${START_EPOCH})..."
    sleep 1   # give the filesystem a moment to flush the folder name

    RAW_NAME=$(find_output_folder "$START_EPOCH")

    if [[ -z "$RAW_NAME" ]]; then
        log "ERROR: could not find a timestamp-named output folder for ${N}T run"
        exit 1
    fi

    RAW_PATH="${OUTPUT_DIR}/${RAW_NAME}"
    log "Found output folder: ${RAW_NAME}"

    # guard against accidentally renaming an already-renamed folder
    if [[ "$RAW_NAME" == physics_second_* ]]; then
        log "ERROR: found folder '${RAW_NAME}' — looks like it was already renamed"
        exit 1
    fi

    mv "$RAW_PATH" "$TARGET_PATH"
    log "Renamed: '${RAW_NAME}'  →  '${TARGET_NAME}'"

    echo ""
    log "Done: ${N}T  |  output: ${TARGET_PATH}"
    echo ""

done

log "Scaling sweep complete."
log "Folders in ${OUTPUT_DIR}:"
ls -1d "${OUTPUT_DIR}"/physics_second_*T_gpuembed 2>/dev/null || \
    log "(none found — check OUTPUT_DIR)"