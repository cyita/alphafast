#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.
#
# AlphaFast run script — NO CONTAINER VERSION (v2)
# Runs Python scripts directly in the current environment.
#
# Changes from v1:
#   - Partition now uses a single Python call (partition_all.py) instead of
#     one process per JSON file — handles 10k+ inputs without hanging.
#   - Relative MSA paths in JSON are resolved to absolute paths during
#     partitioning, so AlphaFold can always find the MSA files.
#   - Phase 2 now uses msa_output/gpu_X/ directly as fold input directories
#     (no fold_partition redistribution step needed).
#
# Usage:
#   ./scripts/run_alphafast_local_v2.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir /path/to/databases \
#       --weights_dir /path/to/weights \
#       [--num_gpus 1] \
#       [--batch_size auto] \
#       [--gpu_devices 0,1,...] \
#       [--app_dir /app/alphafold]

set -euo pipefail

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
DB_DIR=""
WEIGHTS_DIR=""
NUM_GPUS=1
BATCH_SIZE=""
GPU_DEVICES=""
APP_DIR="/app/alphafold"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir DIR --output_dir DIR --db_dir DIR --weights_dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input_dir DIR       Directory containing input JSON files"
    echo "  --output_dir DIR      Output directory for results"
    echo "  --db_dir DIR          Database directory"
    echo "  --weights_dir DIR     Directory containing af3.bin.zst"
    echo ""
    echo "Optional:"
    echo "  --num_gpus N          Number of GPUs (default: 1)"
    echo "  --batch_size N        MSA batch size (default: auto)"
    echo "  --gpu_devices IDS     Comma-separated GPU IDs (default: 0 or 0,1,...,N-1)"
    echo "  --app_dir DIR         Path to alphafold app directory (default: /app/alphafold)"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)    INPUT_DIR="$2";    shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --db_dir)       DB_DIR="$2";       shift 2 ;;
        --weights_dir)  WEIGHTS_DIR="$2";  shift 2 ;;
        --num_gpus)     NUM_GPUS="$2";     shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --gpu_devices)  GPU_DEVICES="$2";  shift 2 ;;
        --app_dir)      APP_DIR="$2";      shift 2 ;;
        --help|-h)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$DB_DIR" ] || [ -z "$WEIGHTS_DIR" ]; then
    echo "ERROR: --input_dir, --output_dir, --db_dir, and --weights_dir are required."
    usage
fi

for d in "$INPUT_DIR" "$DB_DIR" "$WEIGHTS_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Directory not found: $d"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DB_DIR="$(cd "$DB_DIR" && pwd)"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
MMSEQS_DB_DIR="${DB_DIR}/mmseqs"

# Auto batch size
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
    if [ "$BATCH_SIZE" -eq 0 ]; then
        echo "ERROR: No .json files found in $INPUT_DIR"
        exit 1
    fi
fi

# Default GPU devices
if [ -z "$GPU_DEVICES" ]; then
    if [ "$NUM_GPUS" -eq 1 ]; then
        GPU_DEVICES="0"
    else
        GPU_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    fi
fi

# GPU devices to array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "AlphaFast Run (No Container) v2"
echo "=========================================="
echo "App dir:    $APP_DIR"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "DB dir:     $DB_DIR"
echo "MMseqs dir: $MMSEQS_DB_DIR"
echo "Weights:    $WEIGHTS_DIR"
echo "GPUs:       $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size: $BATCH_SIZE"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Single-GPU mode: two-stage pipeline
# ---------------------------------------------------------------------------
if [ "$NUM_GPUS" -eq 1 ]; then
    GPU_ID="${GPU_ARRAY[0]}"

    PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
    INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"

    echo "=== Stage 1: Data Pipeline (MSA search) ==="
    echo "GPU: $GPU_ID | Log: $PIPELINE_LOG"
    echo ""

    CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_data_pipeline.py" \
        --input_dir="$INPUT_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --db_dir="$DB_DIR" \
        --mmseqs_db_dir="$MMSEQS_DB_DIR" \
        --use_mmseqs_gpu \
        --batch_size="$BATCH_SIZE" \
        2>&1 | tee "$PIPELINE_LOG"

    echo ""
    echo "=== Stage 2: Inference ==="
    echo "Log: $INFERENCE_LOG"
    echo ""

    CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
        --input_dir="$OUTPUT_DIR" \
        --model_dir="$WEIGHTS_DIR" \
        --norun_data_pipeline \
        --output_dir="$OUTPUT_DIR" \
        --force_output_dir \
        2>&1 | tee "$INFERENCE_LOG"

# ---------------------------------------------------------------------------
# Multi-GPU mode
# ---------------------------------------------------------------------------
else
    echo "=== Multi-GPU: Phase-Separated Parallel ==="
    echo ""

    MSA_OUTPUT_DIR="${OUTPUT_DIR}/msa_output"
    mkdir -p "$MSA_OUTPUT_DIR"

    # ---- Phase 1: 一次性分区 + 绝对路径转换，然后并行 MSA ----
    echo "--- Partitioning inputs round-robin across $NUM_GPUS GPUs ---"

    # 写出临时分区脚本（一次 Python 进程处理所有文件，避免 10k+ 进程启动开销）
    PARTITION_SCRIPT="/tmp/alphafast_partition_${TIMESTAMP}.py"
    cat > "$PARTITION_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
Round-robin partition input JSONs across N GPU directories.
Resolves relative MSA paths to absolute paths in the process.
"""
import json, sys, pathlib

input_dir   = pathlib.Path(sys.argv[1])
num_gpus    = int(sys.argv[2])
part_prefix = sys.argv[3]   # e.g. /output/partition_

json_files = sorted(input_dir.glob("*.json"))
total = len(json_files)
print(f"Total input files: {total}", flush=True)

# Create partition dirs
part_dirs = []
for i in range(num_gpus):
    d = pathlib.Path(f"{part_prefix}{i}")
    d.mkdir(parents=True, exist_ok=True)
    part_dirs.append(d)

for idx, src in enumerate(json_files):
    dst_dir  = part_dirs[idx % num_gpus]
    data     = json.loads(src.read_text())
    json_dir = src.parent.resolve()

    # Resolve relative MSA / template paths to absolute
    for seq in data.get("sequences", []):
        for chain_type in ("protein", "rna", "dna"):
            chain = seq.get(chain_type)
            if not chain:
                continue
            for key in ("unpairedMsaPath", "pairedMsaPath", "templatesPath"):
                val = chain.get(key)
                if val and not pathlib.Path(val).is_absolute():
                    chain[key] = str((json_dir / val).resolve())

    (dst_dir / src.name).write_text(json.dumps(data, indent=2))

    if (idx + 1) % 1000 == 0:
        print(f"  Partitioned {idx+1}/{total}...", flush=True)

print(f"Done. Distributed {total} files across {num_gpus} partitions.", flush=True)
PYEOF

    python3 "$PARTITION_SCRIPT" "$INPUT_DIR" "$NUM_GPUS" "${OUTPUT_DIR}/partition_"
    echo ""

    # ---- Phase 1: 并行 MSA ----
    echo "--- Phase 1: Parallel MSA (all $NUM_GPUS GPUs) ---"
    MSA_PIDS=()
    for (( i=0; i<NUM_GPUS; i++ )); do
        GPU_ID="${GPU_ARRAY[$i]}"
        PART_INPUT="${OUTPUT_DIR}/partition_${i}"
        PART_MSA_OUT="${MSA_OUTPUT_DIR}/gpu_${i}"
        mkdir -p "$PART_MSA_OUT"
        MSA_LOG="${LOG_DIR}/msa_gpu${GPU_ID}_${TIMESTAMP}.log"

        PART_BATCH=$(find "$PART_INPUT" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
        if [ "$PART_BATCH" -eq 0 ]; then
            echo "  GPU $GPU_ID: no inputs in partition $i, skipping."
            continue
        fi

        echo "  GPU $GPU_ID: $PART_BATCH input(s) → $PART_MSA_OUT | Log: $MSA_LOG"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_data_pipeline.py" \
            --input_dir="$PART_INPUT" \
            --output_dir="$PART_MSA_OUT" \
            --db_dir="$DB_DIR" \
            --mmseqs_db_dir="$MMSEQS_DB_DIR" \
            --use_mmseqs_gpu \
            --batch_size="$PART_BATCH" \
            > "$MSA_LOG" 2>&1 &
        MSA_PIDS+=($!)
    done

    echo ""
    echo "Waiting for all MSA jobs to finish..."
    FAILED=0
    for pid in "${MSA_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: MSA job (PID $pid) failed!"
            FAILED=1
        fi
    done
    if [ "$FAILED" -ne 0 ]; then
        echo "ERROR: One or more MSA jobs failed. Aborting."
        exit 1
    fi
    echo "Phase 1 complete."
    echo ""

    # ---- Phase 2: 直接用 msa_output/gpu_X/ 作为 fold 输入 ----
    # run_data_pipeline.py 输出结构为 msa_output/gpu_X/<name>/<name>_data.json
    # run_alphafold.py 的 --input_dir 接受包含这些子目录的父目录
    echo "--- Phase 2: Parallel Fold (all $NUM_GPUS GPUs) ---"
    FOLD_PIDS=()
    for (( i=0; i<NUM_GPUS; i++ )); do
        GPU_ID="${GPU_ARRAY[$i]}"
        FOLD_INPUT="${MSA_OUTPUT_DIR}/gpu_${i}"
        FOLD_LOG="${LOG_DIR}/fold_gpu${GPU_ID}_${TIMESTAMP}.log"

        FOLD_COUNT=$(find -L "$FOLD_INPUT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        if [ "$FOLD_COUNT" -eq 0 ]; then
            echo "  GPU $GPU_ID: no MSA outputs in gpu_${i}, skipping."
            continue
        fi

        echo "  GPU $GPU_ID: $FOLD_COUNT input(s) | Log: $FOLD_LOG"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
            --input_dir="$FOLD_INPUT" \
            --model_dir="$WEIGHTS_DIR" \
            --norun_data_pipeline \
            --output_dir="$OUTPUT_DIR" \
            --force_output_dir \
            > "$FOLD_LOG" 2>&1 &
        FOLD_PIDS+=($!)
    done

    echo ""
    echo "Waiting for all fold jobs to finish..."
    FAILED=0
    for pid in "${FOLD_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: Fold job (PID $pid) failed!"
            FAILED=1
        fi
    done
    if [ "$FAILED" -ne 0 ]; then
        echo "ERROR: One or more fold jobs failed."
        exit 1
    fi
    echo "Phase 2 complete."
fi

echo ""
echo "=========================================="
echo "AlphaFast Run Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="
