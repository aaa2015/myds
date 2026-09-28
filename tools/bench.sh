#!/usr/bin/env bash
# tools/bench.sh - Reproducible Benchmark Harness for myds
# Conforms to METRIC_FIX_SPEC.md P2-2 specifications.

set -euo pipefail

MODEL=""
PROMPT_FILE=""
PROMPT_TEXT=""
MAX_TOKENS=256
REPEATS=3
JSON_OUTPUT=0
BIN="./bin/myds"

usage() {
    cat <<EOF
Usage: $0 <model.gguf> [options]

Options:
  --prompt-file <path>    Path to prompt file
  --prompt <string>       Prompt string directly
  --max-tokens <int>      Number of tokens to generate (default: 256)
  --repeats <int>         Number of runs for median calculation (default: 3)
  --json                  Output results as machine-readable JSON
  --bin <path>            Path to myds binary (default: ./bin/myds)
  -h, --help              Show this help message
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

MODEL="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --prompt|-p)
            PROMPT_TEXT="$2"
            shift 2
            ;;
        --max-tokens|-n)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --repeats|-r)
            REPEATS="$2"
            shift 2
            ;;
        --json)
            JSON_OUTPUT=1
            shift
            ;;
        --bin)
            BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file '$MODEL' not found" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "Error: binary '$BIN' not executable or not found. Run 'make' first." >&2
    exit 1
fi

PROMPT=""
if [[ -n "$PROMPT_FILE" ]]; then
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo "Error: prompt file '$PROMPT_FILE' not found" >&2
        exit 1
    fi
    PROMPT="$(cat "$PROMPT_FILE")"
elif [[ -n "$PROMPT_TEXT" ]]; then
    PROMPT="$PROMPT_TEXT"
else
    PROMPT="Explain the concept of quantum superposition in three concise sentences."
fi

# Arrays to collect repeat measurements
LOAD_S=()
PREFILL_S=()
TTFT_S=()
PROMPT_TOKENS=()
DECODE_TOKENS=()
DECODE_TPS=()
WALL_S=()
FINISH_REASONS=()

if [[ $JSON_OUTPUT -eq 0 ]]; then
    echo "========================================================="
    echo "Benchmarking myds on Apple Silicon Metal"
    echo "Model      : $MODEL"
    echo "Max Tokens : $MAX_TOKENS"
    echo "Repeats    : $REPEATS"
    echo "========================================================="
fi

for ((r=1; r<=REPEATS; r++)); do
    TMP_OUT=$(mktemp)
    
    # Run greedy inference
    "$BIN" -m "$MODEL" -p "$PROMPT" -n "$MAX_TOKENS" --temp 0.0 > "$TMP_OUT" 2>&1 || {
        echo "Error running benchmark iteration $r" >&2
        cat "$TMP_OUT" >&2
        rm -f "$TMP_OUT"
        exit 1
    }

    # Extract metrics from output line:
    # load X.XX s | prefill X.XX s (prompt X tok, TTFT X.XX s) | decode X tok in X.XX s = X.XX t/s | wall X.XX s
    METRIC_LINE=$(grep -E "load [0-9.]+ s \| prefill" "$TMP_OUT" | tail -n 1 || true)
    
    if [[ -z "$METRIC_LINE" ]]; then
        echo "Warning: could not parse metric line in iteration $r" >&2
        rm -f "$TMP_OUT"
        continue
    fi

    load=$(echo "$METRIC_LINE" | sed -E 's/.*load ([0-9.]+) s \|.*/\1/')
    prefill=$(echo "$METRIC_LINE" | sed -E 's/.*prefill ([0-9.]+) s \(prompt.*/\1/')
    p_tok=$(echo "$METRIC_LINE" | sed -E 's/.*\(prompt ([0-9]+) tok, TTFT.*/\1/')
    ttft=$(echo "$METRIC_LINE" | sed -E 's/.*TTFT ([0-9.]+) s\) \| decode.*/\1/')
    d_tok=$(echo "$METRIC_LINE" | sed -E 's/.*decode ([0-9]+) tok in.*/\1/')
    d_tps=$(echo "$METRIC_LINE" | sed -E 's/.*= ([0-9.]+) t\/s \| wall.*/\1/')
    wall=$(echo "$METRIC_LINE" | sed -E 's/.*wall ([0-9.]+) s/\1/')

    reason="stop"
    if [[ "$d_tok" -ge "$MAX_TOKENS" ]]; then
        reason="length"
    fi

    LOAD_S+=("$load")
    PREFILL_S+=("$prefill")
    TTFT_S+=("$ttft")
    PROMPT_TOKENS+=("$p_tok")
    DECODE_TOKENS+=("$d_tok")
    DECODE_TPS+=("$d_tps")
    WALL_S+=("$wall")
    FINISH_REASONS+=("$reason")

    if [[ $JSON_OUTPUT -eq 0 ]]; then
        echo "Run $r: load=${load}s, prefill=${prefill}s, TTFT=${ttft}s, prompt=${p_tok}tok, decode=${d_tok}tok, tps=${d_tps}t/s, wall=${wall}s"
    fi

    rm -f "$TMP_OUT"
done

COUNT=${#DECODE_TPS[@]}
if [[ $COUNT -eq 0 ]]; then
    echo "Error: No successful runs recorded" >&2
    exit 1
fi

# Compute median index based on DECODE_TPS
# Sort indices by DECODE_TPS values
MEDIAN_IDX=$((COUNT / 2))

# Simple bash selection of median using python/awk if available or sort
SORTED_DATA=$(for ((i=0; i<COUNT; i++)); do
    echo "${DECODE_TPS[$i]} $i"
done | sort -n)

SELECTED_RUN=$(echo "$SORTED_DATA" | sed -n "$((MEDIAN_IDX + 1))p" | awk '{print $2}')

MED_LOAD="${LOAD_S[$SELECTED_RUN]}"
MED_PREFILL="${PREFILL_S[$SELECTED_RUN]}"
MED_TTFT="${TTFT_S[$SELECTED_RUN]}"
MED_PTOK="${PROMPT_TOKENS[$SELECTED_RUN]}"
MED_DTOK="${DECODE_TOKENS[$SELECTED_RUN]}"
MED_TPS="${DECODE_TPS[$SELECTED_RUN]}"
MED_WALL="${WALL_S[$SELECTED_RUN]}"
MED_REASON="${FINISH_REASONS[$SELECTED_RUN]}"

if [[ $JSON_OUTPUT -eq 1 ]]; then
    cat <<EOF
{"model":"$MODEL","prompt_tokens":$MED_PTOK,"max_tokens":$MAX_TOKENS,"load_s":$MED_LOAD,"prefill_s":$MED_PREFILL,"ttft_s":$MED_TTFT,"decode_tokens":$MED_DTOK,"decode_tps":$MED_TPS,"wall_s":$MED_WALL,"finish_reason":"$MED_REASON"}
EOF
else
    echo "---------------------------------------------------------"
    echo "MEDIAN RESULT (Run $((SELECTED_RUN + 1)) of $COUNT):"
    echo "  Prompt Tokens : $MED_PTOK"
    echo "  Decode Tokens : $MED_DTOK"
    echo "  Load Time     : ${MED_LOAD} s"
    echo "  Prefill (TTFT): ${MED_PREFILL} s"
    echo "  Decode Speed  : ${MED_TPS} tokens/s (wall-clock)"
    echo "  Total Wall    : ${MED_WALL} s"
    echo "  Finish Reason : $MED_REASON"
    echo "---------------------------------------------------------"
fi
