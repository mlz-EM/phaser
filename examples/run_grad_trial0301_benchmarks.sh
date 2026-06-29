#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gridsan/mzhu/Tools/STEM_TOOL/phaser/examples"
LOG_DIR="$ROOT/bench_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local code="$1"
    local sigma="$2"
    local weight="${3:-}"
    local gpu="$4"

    local config
    local log
    local py_path

    if [[ "$sigma" == "0" ]]; then
        config="$ROOT/grad_trial0301_${code}_sigma0.json"
        log="$LOG_DIR/${code}_sigma0.log"
    else
        local weight_tag="${weight/./p}"
        config="$ROOT/grad_trial0301_${code}_sigma${sigma}_w${weight_tag}.json"
        log="$LOG_DIR/${code}_sigma${sigma}_w${weight_tag}.log"
    fi

    case "$code" in
        main)
            py_path="$ROOT/bench_code_main"
            ;;
        pr52)
            py_path="$ROOT/bench_code_pr52"
            ;;
        *)
            echo "Unknown code label: $code" >&2
            return 1
            ;;
    esac

    (
        cd "$ROOT"
        echo "START code=$code sigma=$sigma weight=$weight gpu=$gpu config=$config"
        CUDA_DEVICE="$gpu" CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$py_path" \
            python -m phaser run "$config"
    ) 2>&1 | tee "$log"
}

run_pair() {
    local sigma="$1"
    local weight="${2:-}"

    run_one main "$sigma" "$weight" 0 &
    local pid0=$!
    run_one pr52 "$sigma" "$weight" 1 &
    local pid1=$!

    wait "$pid0"
    wait "$pid1"
}

run_pair 200 0.1
run_pair 150 0.2
run_pair 100 0.3
run_pair 50 0.5
run_pair 0
