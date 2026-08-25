#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.
#
# AlphaFast run script — NO CONTAINER VERSION (v4)
#
# Changes from v3:
#   - Phase 2 supports resume: already-completed jobs (those with *_model.cif
#     in their output directory) are skipped automatically. Re-running the
#     same command after an interruption will only process the remaining jobs.
#
# Usage:
#   ./scripts/run_alphafast_local_v4.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       [--db_dir ...] [--mmseqs_db_dir ...] [--weights_dir ...] \
#       [--num_gpus 4] [--batch_size 500] \
#       [--gpu_devices 0,1,...] [--app_dir /path/to/alphafast]

set -euo pipefail

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
DB_DIR="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/genetic_databases"
MMSEQS_DB_DIR="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/mmseq2"
WEIGHTS_DIR="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/Ruiqi_Lin/code/af3/weight"
NUM_GPUS=1
BATCH_SIZE=""
AUTO_BATCH_SIZE_MODE=""
GPU_DEVICES=""
APP_DIR="/app/alphafold"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir DIR --output_dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input_dir DIR       Directory containing input JSON files"
    echo "  --output_dir DIR      Output directory for results"
    echo ""
    echo "Optional:"
    echo "  --db_dir DIR          Database directory (default: $DB_DIR)"
    echo "  --mmseqs_db_dir DIR   MMseqs2 database directory (default: $MMSEQS_DB_DIR)"
    echo "  --weights_dir DIR     Directory containing af3.bin.zst (default: $WEIGHTS_DIR)"
    echo "  --num_gpus N          Number of GPUs (default: 1)"
    echo "  --batch_size N        MSA batch size (default: auto)"
    echo "  --gpu_devices IDS     Comma-separated GPU IDs (default: 0 or 0,1,...,N-1)"
    echo "  --app_dir DIR         Path to alphafast repo (default: /app/alphafold)"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)      INPUT_DIR="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --db_dir)         DB_DIR="$2";         shift 2 ;;
        --mmseqs_db_dir)  MMSEQS_DB_DIR="$2";  shift 2 ;;
        --weights_dir)    WEIGHTS_DIR="$2";    shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";       shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --gpu_devices)    GPU_DEVICES="$2";    shift 2 ;;
        --app_dir)        APP_DIR="$2";        shift 2 ;;
        --help|-h)        usage ;;
        *)                echo "Unknown argument: $1"; usage ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --input_dir and --output_dir are required."
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

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PHASE1_DONE="${OUTPUT_DIR}/.phase1_done"
PHASE2_DONE="${OUTPUT_DIR}/.phase2_done"

# Auto batch size
if [ -z "$BATCH_SIZE" ]; then
    INPUT_JSON_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
    if [ "$INPUT_JSON_COUNT" -eq 0 ]; then
        echo "ERROR: No .json files found in $INPUT_DIR"
        exit 1
    fi
    if [ "$NUM_GPUS" -gt 1 ]; then
        BATCH_SIZE=256
        AUTO_BATCH_SIZE_MODE="auto-multi-gpu-capped"
    else
        BATCH_SIZE="$INPUT_JSON_COUNT"
        AUTO_BATCH_SIZE_MODE="auto-single-gpu-all-inputs"
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

IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"

echo "=========================================="
echo "AlphaFast Run (No Container) v4"
echo "=========================================="
echo "App dir:      $APP_DIR"
echo "Input dir:    $INPUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "DB dir:       $DB_DIR"
echo "MMseqs dir:   $MMSEQS_DB_DIR"
echo "Weights:      $WEIGHTS_DIR"
echo "GPUs:         $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size:   $BATCH_SIZE"
if [ -n "$AUTO_BATCH_SIZE_MODE" ]; then
    echo "Batch mode:   $AUTO_BATCH_SIZE_MODE"
fi
echo "Log dir:      $LOG_DIR"
echo "Start time:   $(date)"
echo "=========================================="
echo ""

# Check if already fully done
if [ -f "$PHASE2_DONE" ]; then
    echo "Already fully complete (.phase2_done exists). Nothing to do."
    echo "Delete $PHASE2_DONE to force re-run."
    exit 0
fi

# ---------------------------------------------------------------------------
# Single-GPU mode
# ---------------------------------------------------------------------------
if [ "$NUM_GPUS" -eq 1 ]; then
    GPU_ID="${GPU_ARRAY[0]}"

    if [ ! -f "$PHASE1_DONE" ]; then
        PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
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

        touch "$PHASE1_DONE"
        echo "Phase 1 complete. Marker written: $PHASE1_DONE"
    else
        echo "=== Stage 1: Skipping (phase1_done marker exists) ==="
    fi

    echo ""
    INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"
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

    touch "$PHASE2_DONE"
    echo "Phase 2 complete. Marker written: $PHASE2_DONE"

# ---------------------------------------------------------------------------
# Multi-GPU mode
# ---------------------------------------------------------------------------
else
    MSA_OUTPUT_DIR="${OUTPUT_DIR}/msa_output"

    # ---- Phase 1 ----
    if [ ! -f "$PHASE1_DONE" ]; then
        echo "=== Multi-GPU: Phase-Separated Parallel ==="
        echo ""
        mkdir -p "$MSA_OUTPUT_DIR"

        PARTITION_SCRIPT="/tmp/alphafast_partition_${TIMESTAMP}.py"
        cat > "$PARTITION_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json, sys, pathlib

input_dir   = pathlib.Path(sys.argv[1])
num_gpus    = int(sys.argv[2])
part_prefix = sys.argv[3]

json_files = sorted(input_dir.glob("*.json"))
total = len(json_files)
print(f"Total input files: {total}", flush=True)

part_dirs = []
for i in range(num_gpus):
    d = pathlib.Path(f"{part_prefix}{i}")
    d.mkdir(parents=True, exist_ok=True)
    part_dirs.append(d)

for idx, src in enumerate(json_files):
    dst_dir  = part_dirs[idx % num_gpus]
    data     = json.loads(src.read_text())
    json_dir = src.parent.resolve()

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

        echo "--- Partitioning inputs round-robin across $NUM_GPUS GPUs ---"
        python3 "$PARTITION_SCRIPT" "$INPUT_DIR" "$NUM_GPUS" "${OUTPUT_DIR}/partition_"
        echo ""

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

            PART_MSA_BATCH="$BATCH_SIZE"
            if [ "$PART_BATCH" -lt "$PART_MSA_BATCH" ]; then
                PART_MSA_BATCH="$PART_BATCH"
            fi

            echo "  GPU $GPU_ID: $PART_BATCH input(s), initial_msa_batch=$PART_MSA_BATCH → $PART_MSA_OUT | Log: $MSA_LOG"
            (
                attempt=1
                current_batch="$PART_MSA_BATCH"
                while true; do
                    rm -rf "$PART_MSA_OUT"
                    mkdir -p "$PART_MSA_OUT"
                    if [ "$attempt" -eq 1 ]; then
                        printf '[attempt %d] gpu=%s inputs=%s batch_size=%s\n' "$attempt" "$GPU_ID" "$PART_BATCH" "$current_batch" > "$MSA_LOG"
                    else
                        printf '\n[attempt %d] gpu=%s retry with smaller batch_size=%s\n' "$attempt" "$GPU_ID" "$current_batch" >> "$MSA_LOG"
                    fi

                    if CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_data_pipeline.py" \
                        --input_dir="$PART_INPUT" \
                        --output_dir="$PART_MSA_OUT" \
                        --db_dir="$DB_DIR" \
                        --mmseqs_db_dir="$MMSEQS_DB_DIR" \
                        --use_mmseqs_gpu \
                        --batch_size="$current_batch" \
                        >> "$MSA_LOG" 2>&1; then
                        printf '\n[attempt %d] gpu=%s succeeded with batch_size=%s\n' "$attempt" "$GPU_ID" "$current_batch" >> "$MSA_LOG"
                        exit 0
                    fi

                    if [ "$current_batch" -le 1 ]; then
                        echo "ERROR: GPU $GPU_ID MSA failed even at batch_size=$current_batch" | tee -a "$MSA_LOG"
                        exit 1
                    fi

                    next_batch=$(( (current_batch + 1) / 2 ))
                    if [ "$next_batch" -ge "$current_batch" ]; then
                        next_batch=$(( current_batch - 1 ))
                    fi
                    echo "WARN: GPU $GPU_ID MSA failed at batch_size=$current_batch; retrying with batch_size=$next_batch" | tee -a "$MSA_LOG"
                    current_batch="$next_batch"
                    attempt=$(( attempt + 1 ))
                done
            ) &
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

        touch "$PHASE1_DONE"
        echo "Phase 1 complete. Marker written: $PHASE1_DONE"
        echo ""
    else
        echo "=== Phase 1: Skipping (phase1_done marker exists) ==="
        echo ""
    fi

    # ---- Phase 2: 支持断点续跑，跳过已完成的 job ----
    echo "--- Phase 2: Parallel Fold (all $NUM_GPUS GPUs) ---"

    # 为每个 GPU 建临时的 resume 输入目录，只放未完成的 job
    RESUME_SCRIPT="/tmp/alphafast_resume_${TIMESTAMP}.py"
    cat > "$RESUME_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
For each gpu_X dir under msa_output, check which jobs are already done
(have *_model.cif in output_dir/<job_name>/). Build per-GPU resume dirs
containing only symlinks to incomplete jobs.
Prints: gpu_index  resume_dir  todo_count
"""
import sys, pathlib, json, shutil

msa_output_dir = pathlib.Path(sys.argv[1])
output_dir     = pathlib.Path(sys.argv[2])
resume_base    = pathlib.Path(sys.argv[3])
num_gpus       = int(sys.argv[4])

total_done = 0
total_todo = 0

for i in range(num_gpus):
    gpu_dir    = msa_output_dir / f"gpu_{i}"
    resume_dir = resume_base / f"gpu_{i}"
    # Clean and recreate resume dir
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    resume_dir.mkdir(parents=True)

    todo = 0
    done = 0
    for job_dir in sorted(gpu_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        out_dir = output_dir / job_dir.name
        if list(out_dir.glob("*_model.cif")):
            done += 1
        else:
            # Symlink the job dir into resume dir
            (resume_dir / job_dir.name).symlink_to(job_dir.resolve())
            todo += 1

    total_done += done
    total_todo += todo
    print(f"{i}\t{resume_dir}\t{todo}", flush=True)

print(f"SUMMARY\t-\tDone: {total_done}, Todo: {total_todo}", flush=True)
PYEOF

    RESUME_BASE="${OUTPUT_DIR}/.resume_dirs"
    mkdir -p "$RESUME_BASE"

    # Run resume script and capture per-GPU info
    RESUME_OUTPUT=$(python3 "$RESUME_SCRIPT" \
        "$MSA_OUTPUT_DIR" "$OUTPUT_DIR" "$RESUME_BASE" "$NUM_GPUS")

    echo "$RESUME_OUTPUT"
    echo ""

    FOLD_PIDS=()
    for (( i=0; i<NUM_GPUS; i++ )); do
        GPU_ID="${GPU_ARRAY[$i]}"
        RESUME_DIR="${RESUME_BASE}/gpu_${i}"
        FOLD_LOG="${LOG_DIR}/fold_gpu${GPU_ID}_${TIMESTAMP}.log"

        FOLD_COUNT=$(find -L "$RESUME_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        if [ "$FOLD_COUNT" -eq 0 ]; then
            echo "  GPU $GPU_ID: all jobs already done, skipping."
            continue
        fi

        echo "  GPU $GPU_ID: $FOLD_COUNT job(s) remaining | Log: $FOLD_LOG"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
            --input_dir="$RESUME_DIR" \
            --model_dir="$WEIGHTS_DIR" \
            --norun_data_pipeline \
            --output_dir="$OUTPUT_DIR" \
            --force_output_dir \
            > "$FOLD_LOG" 2>&1 &
        FOLD_PIDS+=($!)
    done

    if [ ${#FOLD_PIDS[@]} -eq 0 ]; then
        echo "All fold jobs already complete!"
    else
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
    fi

    touch "$PHASE2_DONE"
    echo "Phase 2 complete. Marker written: $PHASE2_DONE"
fi

echo ""
echo "=========================================="
echo "AlphaFast Run Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="
