#!/bin/bash
# run_all_plots.sh
# Usage: bash run_all_plots.sh [output_dir]
# Discovers all experiment subfolders automatically (any folder containing
# Batch_*_Query_*_stats.csv files) and saves plots to <folder>/plots/
# Defaults to the current directory if no argument given.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_rerank.py"
OUTPUT_DIR="${1:-.}"

found=0
for dir in "$OUTPUT_DIR"/*/; do
    # skip any "plots" subdirectory
    [[ "$(basename "$dir")" == "plots" ]] && continue
    # only process folders that actually contain stats CSVs
    if ls "$dir"Batch_*_Query_*_stats.csv &>/dev/null 2>&1; then
        echo "=== $(basename "$dir") ==="
        python "$PLOT_SCRIPT" "$dir" --save-dir "$dir/plots"
        ((found++))
    fi
done

echo ""
if [ "$found" -eq 0 ]; then
    echo "No experiment folders found in: $OUTPUT_DIR"
else
    echo "Done – processed $found folder(s). Plots saved in <folder>/plots/"
fi