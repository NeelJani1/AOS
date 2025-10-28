#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT_PATH="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification/evaluate_model.py"
BASE_DIR="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification/results/adaptive_otsu_fixed_all_20251026_022500/unlearning"
OUTPUT_CSV="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification/results/adaptive_otsu_fixed_all_20251026_022500/evaluation/re-evaluated_results.csv"
# --- End of Configuration ---

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "=== RUNNING ALL EVALUATIONS ==="
echo "Saving results to: $OUTPUT_CSV"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "🚨 ERROR: Cannot find evaluation script at:"
    echo "   $PYTHON_SCRIPT_PATH"
    exit 1
fi

# Create CSV Header
echo "method,percentage,test_acc,forget_acc,retain_acc,model_path" > "$OUTPUT_CSV"

# This function calls your python script and gets the result
get_results() {
    local method=$1
    local model_file=$2
    local percent_int=$3 # e.g., 10, 20
    local percent_float="0.${percent_int}" # e.g., 0.10, 0.20

    # Handle 100%
    if [ "$percent_int" == "100" ]; then
        percent_float="1.0"
    fi

    echo "   Evaluating: $method $percent_int%..."

    # Run the Python script and capture all its output
    local output_all=$(python3 "$PYTHON_SCRIPT_PATH" --model_path "$model_file" --forget_perc "$percent_float")

    # ### <<< NEW FIX HERE ###
    # Grab only the *last line* of the output, which is the CSV
    local output_csv=$(echo "$output_all" | tail -1)

    if [ -z "$output_csv" ] || [[ "$output_csv" == *"Error"* ]] || [[ "$output_csv" != *","* ]]; then
        echo "      ❌ FAILED. Full output:"
        echo "$output_all"
        echo "$method,$percent_int,ERROR,ERROR,ERROR,$model_file" >> "$OUTPUT_CSV"
    else
        # Now, parse the clean CSV line
        local forget_acc=$(echo "$output_csv" | cut -d',' -f1)
        local retain_acc=$(echo "$output_csv" | cut -d',' -f2)
        local test_acc=$(echo "$output_csv" | cut -d',' -f3)

        echo "      ✅ Test: $test_acc% | Forget: $forget_acc% | Retain: $retain_acc%"
        echo "$method,$percent_int,$test_acc,$forget_acc,$retain_acc,$model_file" >> "$OUTPUT_CSV"
    fi
}

# ---
# ROBUST FIND COMMAND
# ---
echo ""
echo "🔍 Searching for all model_SA_best.pth.tar files..."

find "$BASE_DIR" -type f -name "model_SA_best.pth.tar" | sort -V | while read model_file; do

    dir=$(dirname "$model_file")
    run_name=$(basename "$dir")

    method=""
    if [[ "$run_name" == *"_GA_"* ]]; then
        method="GA"
    elif [[ "$run_name" == *"_FT_"* ]]; then
        method="FT"
    elif [[ "$run_name" == *"_RL_"* ]]; then
        method="RL"
    else
        echo "   (Skipping unknown folder structure: $dir)"
        continue
    fi

    percent=$(echo "$run_name" | grep -o '[0-9]\+percent' | head -1 | sed 's/percent//')

    if [ -z "$percent" ]; then
        echo "   (Skipping folder with no percentage: $dir)"
        continue
    fi

    get_results "$method" "$model_file" "$percent"
done

echo ""
echo "=== EVALUATION COMPLETE ==="
echo "✅ All results saved to: $OUTPUT_CSV"
