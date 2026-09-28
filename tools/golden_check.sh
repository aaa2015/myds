#!/usr/bin/env bash
# tools/golden_check.sh - Deterministic behavior regression comparison ONLY.
# Matching an old myds binary cannot establish model correctness. Numerical
# semantics fixes require an independent reference and tensor/logit comparison.
# Conforms to METRIC_FIX_SPEC.md P0-GATE specifications.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <old_bin> <new_bin> [model.gguf]

Parameters:
  <old_bin>     Path to baseline myds binary
  <new_bin>     Path to candidate myds binary
  [model.gguf]  Optional path to GGUF model (default: auto-detect or env MODEL_PATH)

Verification:
  - BEHAVIOR ONLY: this is not a model correctness or generation-quality gate
  - Do not use it to reject a numerical fix merely because broken output changes
  - Runs >= 3 diverse prompts at temperature=0.0 (greedy) with fixed seed=42
  - Includes at least 1 deterministic verbatim recall prompt
  - Verifies bit-exact token ID sequence equivalence via SHA256 hashes
EOF
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

OLD_BIN="$1"
NEW_BIN="$2"
MODEL="${3:-${MODEL_PATH:-}}"

if [[ ! -x "$OLD_BIN" ]]; then
    echo "Error: old_bin '$OLD_BIN' not executable" >&2
    exit 1
fi

if [[ ! -x "$NEW_BIN" ]]; then
    echo "Error: new_bin '$NEW_BIN' not executable" >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    CANDIDATES=(
        "~/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2.gguf"
        "/tmp/test_myds_mla_forward.gguf"
    )
    for c in "${CANDIDATES[@]}"; do
        if [[ -f "$c" ]]; then
            MODEL="$c"
            break
        fi
    done
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file '$MODEL' not found" >&2
    exit 1
fi

echo "========================================================="
echo "  Behavior regression comparison (NOT a correctness oracle)"
echo "========================================================="
echo "Baseline Bin  : $OLD_BIN"
echo "Candidate Bin : $NEW_BIN"
echo "Model         : $MODEL"
echo "========================================================="

PROMPTS=(
    "Repeat the following text verbatim: DeepSeek-V4.1-Flash is an open-source Mixture-of-Experts language model."
    "Calculate 127 * 43 step by step and provide the final answer."
    "Write a concise C function that computes the square of an integer."
)

ALL_PASSED=1

echo "--- Suite 1: Greedy (temp=0.0, seed=42) [HARD GATE] ---"
for idx in "${!PROMPTS[@]}"; do
    p_num=$((idx + 1))
    p_text="${PROMPTS[$idx]}"
    
    echo -n "Test Prompt $p_num: \"${p_text:0:50}...\" -> "

    TMP_OLD=$(mktemp)
    TMP_NEW=$(mktemp)

    "$OLD_BIN" -m "$MODEL" -p "$p_text" -n 64 --temp 0.0 --seed 42 --print-tokens > "$TMP_OLD" 2>&1 || {
        echo "FAIL (old_bin execution error)"
        cat "$TMP_OLD" >&2
        rm -f "$TMP_OLD" "$TMP_NEW"
        exit 1
    }

    "$NEW_BIN" -m "$MODEL" -p "$p_text" -n 64 --temp 0.0 --seed 42 --print-tokens > "$TMP_NEW" 2>&1 || {
        echo "FAIL (new_bin execution error)"
        cat "$TMP_NEW" >&2
        rm -f "$TMP_OLD" "$TMP_NEW"
        exit 1
    }

    OLD_TOKENS=$(grep -E "^TOKEN_IDS: " "$TMP_OLD" | head -n 1 || true)
    NEW_TOKENS=$(grep -E "^TOKEN_IDS: " "$TMP_NEW" | head -n 1 || true)

    if [[ -z "$OLD_TOKENS" || -z "$NEW_TOKENS" ]]; then
        echo "FAIL (TOKEN_IDS line missing from output)"
        ALL_PASSED=0
        rm -f "$TMP_OLD" "$TMP_NEW"
        continue
    fi

    OLD_HASH=$(echo "$OLD_TOKENS" | shasum -a 256 | awk '{print $1}')
    NEW_HASH=$(echo "$NEW_TOKENS" | shasum -a 256 | awk '{print $1}')

    if [[ "$OLD_HASH" == "$NEW_HASH" ]]; then
        echo "PASS [sha256=${OLD_HASH:0:16}...]"
    else
        echo "FAIL"
        echo "  Baseline Hash : $OLD_HASH"
        echo "  Candidate Hash: $NEW_HASH"
        echo "  Diff in token IDs:"
        diff -u <(echo "$OLD_TOKENS") <(echo "$NEW_TOKENS") || true
        ALL_PASSED=0
    fi

    rm -f "$TMP_OLD" "$TMP_NEW"
done

echo ""
echo "--- Suite 2: Stochastic Sampling (temp=0.7, seed=42) [EXTENDED GATE] ---"

run_sampling_check() {
    local label="$1"
    local top_k_arg="$2"
    local prompt="Write a concise C function that computes the square of an integer."

    echo -n "Sampling Test ($label, top-k=$top_k_arg) -> "

    TMP_OLD=$(mktemp)
    TMP_NEW=$(mktemp)

    "$OLD_BIN" -m "$MODEL" -p "$prompt" -n 64 --temp 0.7 --seed 42 --top-k "$top_k_arg" --print-tokens > "$TMP_OLD" 2>&1 || {
        echo "FAIL (old_bin execution error)"
        cat "$TMP_OLD" >&2
        rm -f "$TMP_OLD" "$TMP_NEW"
        return 1
    }

    "$NEW_BIN" -m "$MODEL" -p "$prompt" -n 64 --temp 0.7 --seed 42 --top-k "$top_k_arg" --print-tokens > "$TMP_NEW" 2>&1 || {
        echo "FAIL (new_bin execution error)"
        cat "$TMP_NEW" >&2
        rm -f "$TMP_OLD" "$TMP_NEW"
        return 1
    }

    OLD_TOKENS=$(grep -E "^TOKEN_IDS: " "$TMP_OLD" | head -n 1 || true)
    NEW_TOKENS=$(grep -E "^TOKEN_IDS: " "$TMP_NEW" | head -n 1 || true)

    if [[ -z "$OLD_TOKENS" || -z "$NEW_TOKENS" ]]; then
        echo "FAIL (TOKEN_IDS line missing from output)"
        rm -f "$TMP_OLD" "$TMP_NEW"
        ALL_PASSED=0
        return 1
    fi

    OLD_HASH=$(echo "$OLD_TOKENS" | shasum -a 256 | awk '{print $1}')
    NEW_HASH=$(echo "$NEW_TOKENS" | shasum -a 256 | awk '{print $1}')

    if [[ "$OLD_HASH" == "$NEW_HASH" ]]; then
        echo "PASS [sha256=${OLD_HASH:0:16}...]"
    else
        echo "DIVERGENCE"
        echo "  Baseline Hash : $OLD_HASH"
        echo "  Candidate Hash: $NEW_HASH"
        echo "  Diff in token IDs:"
        diff -u <(echo "$OLD_TOKENS") <(echo "$NEW_TOKENS") || true
        # If temp>0 differs, record as warning/investigation point unless explicitly accepted
        ALL_PASSED=0
    fi

    rm -f "$TMP_OLD" "$TMP_NEW"
}

run_sampling_check "top-k=0 (pure softmax)" 0 || true
run_sampling_check "top-k=40 (standard pool)" 40 || true

echo "========================================================="
if [[ $ALL_PASSED -eq 1 ]]; then
    echo "  Token ID sequences match. Model correctness remains unverified."
    echo "========================================================="
    exit 0
else
    echo "  GOLDEN CHECK FAILED! Token ID divergence detected."
    echo "========================================================="
    exit 1
fi
