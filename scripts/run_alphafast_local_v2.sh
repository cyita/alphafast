#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.
#
# AlphaFast run script — NO CONTAINER VERSION (v3)
#
# Changes from v2:
#   - LOG_DIR is now absolute (under OUTPUT_DIR), safe to run from any directory.
#   - Phase 1 writes .phase1_done marker on completion; subsequent runs skip
#     Phase 1 automatically if the marker exists.
#   - Phase 2 writes .phase2_done marker on completion.
#   - All directory paths are now required arguments (no hardcoded defaults).
#   - APP_DIR is auto-derived from the script's own location (scripts/../),
#     no longer needed as a user-facing argument.
#
# v2 modifications:
#   - Added --mmseqs_n_threads parameter to limit thread count per GPU process
#   - Default: 15 threads per process (120 CPU cores / 8 GPUs)
#   - Prevents thread oversubscription that can cause CUDA initialization failures
#
# Usage:
#   bash /path/to/alphafast/scripts/run_alphafast_local_v2.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir /path/to/databases \
#       --mmseqs_db_dir /path/to/mmseqs \
#       --weights_dir /path/to/weights \
#       [--num_gpus 1] \
#       [--batch_size 500] \
#       [--gpu_devices 0,1,...] \
#       [--mmseqs_threads 15]

set -euo pipefail

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
DB_DIR=""
MMSEQS_DB_DIR=""
WEIGHTS_DIR=""
TEMP_DIR=""
NUM_GPUS=1
BATCH_SIZE=""
AUTO_BATCH_SIZE_MODE=""
GPU_DEVICES=""
NUM_SEEDS=3
MMSEQS_THREADS=15  # Default: 15 threads per process
TEMPLATE_BATCH_SIZE=""
TEMPLATE_MAX_ATTEMPTS=""
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir DIR --output_dir DIR --db_dir DIR --mmseqs_db_dir DIR --weights_dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input_dir DIR       Directory containing input JSON files"
    echo "  --output_dir DIR      Output directory for results"
    echo "  --db_dir DIR          Genetic database directory"
    echo "  --mmseqs_db_dir DIR   MMseqs2 database directory"
    echo "  --weights_dir DIR     Directory containing af3.bin.zst"
    echo ""
    echo "Optional:"
    echo "  --temp_dir DIR        Temporary directory for MMseqs work files"
    echo "  --num_gpus N          Number of GPUs (default: 1)"
    echo "  --batch_size N        MSA batch size (default: auto)"
    echo "  --gpu_devices IDS     Comma-separated GPU IDs (default: 0 or 0,1,...,N-1)"
    echo "  --num_seeds N         Number of inference seeds (default: 3)"
    echo "  --mmseqs_threads N    MMseqs2 threads per GPU (default: 15)"
    echo "  --template_batch_size N   Template batch size (default: auto)"
    echo "  --template_max_attempts N Template retry attempts (default: auto)"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)      INPUT_DIR="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --db_dir)         DB_DIR="$2";         shift 2 ;;
        --mmseqs_db_dir)  MMSEQS_DB_DIR="$2";  shift 2 ;;
        --weights_dir)    WEIGHTS_DIR="$2";    shift 2 ;;
        --temp_dir)       TEMP_DIR="$2";       shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";       shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --gpu_devices)    GPU_DEVICES="$2";    shift 2 ;;
        --num_seeds)      NUM_SEEDS="$2";      shift 2 ;;
        --mmseqs_threads) MMSEQS_THREADS="$2"; shift 2 ;;
        --template_batch_size) TEMPLATE_BATCH_SIZE="$2"; shift 2 ;;
        --template_max_attempts) TEMPLATE_MAX_ATTEMPTS="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *)                echo "Unknown argument: $1"; usage ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$DB_DIR" ] || [ -z "$MMSEQS_DB_DIR" ] || [ -z "$WEIGHTS_DIR" ]; then
    echo "ERROR: --input_dir, --output_dir, --db_dir, --mmseqs_db_dir, and --weights_dir are all required."
    usage
fi

for d in "$INPUT_DIR" "$DB_DIR" "$WEIGHTS_DIR" "$APP_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Directory not found: $d"
        exit 1
    fi
done

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${APP_DIR}/src:${PYTHONPATH}"
else
    export PYTHONPATH="${APP_DIR}/src"
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DB_DIR="$(cd "$DB_DIR" && pwd)"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"

# LOG_DIR is absolute, always under OUTPUT_DIR
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Phase marker files
PHASE1_DONE="${OUTPUT_DIR}/.phase1_done"
PHASE2_DONE="${OUTPUT_DIR}/.phase2_done"
if [ -z "$TEMP_DIR" ]; then
    TEMP_DIR="${OUTPUT_DIR}/temp_${TIMESTAMP}"
fi
mkdir -p "$TEMP_DIR"
TEMP_DIR="$(cd "$TEMP_DIR" && pwd)"

PREPARE_SINGLE_SEED_SCRIPT="${TEMP_DIR}/prepare_phase2_single_seed_inputs.py"
cat > "$PREPARE_SINGLE_SEED_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json
import pathlib
import shutil
import sys


src_root = pathlib.Path(sys.argv[1])
dst_root = pathlib.Path(sys.argv[2])

if dst_root.exists():
    shutil.rmtree(dst_root)
dst_root.mkdir(parents=True, exist_ok=True)

prepared = 0
for data_json_path in sorted(src_root.glob("*/*_data.json")):
    src_job_dir = data_json_path.parent
    dst_job_dir = dst_root / src_job_dir.name
    dst_job_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(src_job_dir.iterdir()):
        if child.name == data_json_path.name:
            continue
        (dst_job_dir / child.name).symlink_to(child.resolve())

    raw = json.loads(data_json_path.read_text())
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"{data_json_path} contains {len(raw)} jobs; expected one job per data JSON."
            )
        fold_job = raw[0]
        wrapped = True
    else:
        fold_job = raw
        wrapped = False

    model_seeds = fold_job.get("modelSeeds")
    if not model_seeds:
        raise ValueError(f"{data_json_path} does not contain any modelSeeds.")
    fold_job["modelSeeds"] = [model_seeds[0]]

    payload = [fold_job] if wrapped else fold_job
    (dst_job_dir / data_json_path.name).write_text(json.dumps(payload, indent=2))
    prepared += 1

print(f"Prepared {prepared} single-seed phase-2 input(s) under {dst_root}", flush=True)
PYEOF
chmod +x "$PREPARE_SINGLE_SEED_SCRIPT"

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

# GPU devices to array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"

echo "=========================================="
echo "AlphaFast Run (Native, No Container) v2"
echo "=========================================="
echo "App dir:      $APP_DIR"
echo "Input dir:    $INPUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "DB dir:       $DB_DIR"
echo "MMseqs dir:   $MMSEQS_DB_DIR"
echo "Weights:      $WEIGHTS_DIR"
echo "Temp dir:     $TEMP_DIR"
echo "GPUs:         $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size:   $BATCH_SIZE"
if [ -n "$AUTO_BATCH_SIZE_MODE" ]; then
    echo "Batch mode:   $AUTO_BATCH_SIZE_MODE"
fi
echo "Num seeds:    $NUM_SEEDS"
echo "MMseqs threads: $MMSEQS_THREADS per GPU"
echo "Log dir:      $LOG_DIR"
echo "Start time:   $(date)"
echo "=========================================="
echo ""

# Check if already fully done
if [ -f "$PHASE2_DONE" ]; then
    echo "Phase 2 already completed (.phase2_done exists). Nothing to do."
    echo "Delete $PHASE2_DONE to force re-run."
    exit 0
fi

# ---------------------------------------------------------------------------
# Single-GPU mode: two-stage pipeline
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
            --mmseqs_n_threads="$MMSEQS_THREADS" \
            --temp_dir="${TEMP_DIR}/gpu_${GPU_ID}" \
            ${TEMPLATE_BATCH_SIZE:+--template_batch_size="$TEMPLATE_BATCH_SIZE"} \
            ${TEMPLATE_MAX_ATTEMPTS:+--template_max_attempts="$TEMPLATE_MAX_ATTEMPTS"} \
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

    if [ "$NUM_SEEDS" -eq 1 ]; then
        PHASE2_INPUT_DIR="${TEMP_DIR}/phase2_single_seed_inputs"
        (
            python3 "$PREPARE_SINGLE_SEED_SCRIPT" "$OUTPUT_DIR" "$PHASE2_INPUT_DIR"
            CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
                --input_dir="$PHASE2_INPUT_DIR" \
                --model_dir="$WEIGHTS_DIR" \
                --norun_data_pipeline \
                --output_dir="$OUTPUT_DIR" \
                --force_output_dir
        ) 2>&1 | tee "$INFERENCE_LOG"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
            --input_dir="$OUTPUT_DIR" \
            --model_dir="$WEIGHTS_DIR" \
            --norun_data_pipeline \
            --num_seeds="$NUM_SEEDS" \
            --output_dir="$OUTPUT_DIR" \
            --force_output_dir \
            2>&1 | tee "$INFERENCE_LOG"
    fi
    
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
        
        # Write partition script inline
        PARTITION_SCRIPT="${TEMP_DIR}/partition_${TIMESTAMP}.py"
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

            echo "  GPU $GPU_ID: $PART_BATCH input(s), initial_msa_batch=$PART_MSA_BATCH → $PART_MSA_OUT | Log: $MSA_LOG | Threads: $MMSEQS_THREADS"
            (
                mkdir -p "$PART_MSA_OUT"
                CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_data_pipeline.py" \
                    --input_dir="$PART_INPUT" \
                    --output_dir="$PART_MSA_OUT" \
                    --db_dir="$DB_DIR" \
                    --mmseqs_db_dir="$MMSEQS_DB_DIR" \
                    --use_mmseqs_gpu \
                    --batch_size="$PART_MSA_BATCH" \
                    --mmseqs_n_threads="$MMSEQS_THREADS" \
                    --temp_dir="${TEMP_DIR}/gpu_${GPU_ID}" \
                    ${TEMPLATE_BATCH_SIZE:+--template_batch_size="$TEMPLATE_BATCH_SIZE"} \
                    ${TEMPLATE_MAX_ATTEMPTS:+--template_max_attempts="$TEMPLATE_MAX_ATTEMPTS"} \
                    >> "$MSA_LOG" 2>&1
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
            echo "Fix the error and re-run; Phase 1 will restart from scratch."
            exit 1
        fi
        
        touch "$PHASE1_DONE"
        echo "Phase 1 complete. Marker written: $PHASE1_DONE"
        echo ""
    else
        echo "=== Phase 1: Skipping (phase1_done marker exists) ==="
        echo ""
    fi
    
    # ---- Phase 2 ----
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
        if [ "$NUM_SEEDS" -eq 1 ]; then
            FOLD_PHASE2_INPUT="${TEMP_DIR}/phase2_single_seed/gpu_${i}"
            (
                python3 "$PREPARE_SINGLE_SEED_SCRIPT" "$FOLD_INPUT" "$FOLD_PHASE2_INPUT"
                CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
                    --input_dir="$FOLD_PHASE2_INPUT" \
                    --model_dir="$WEIGHTS_DIR" \
                    --norun_data_pipeline \
                    --output_dir="$OUTPUT_DIR" \
                    --force_output_dir
            ) > "$FOLD_LOG" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="$GPU_ID" python "${APP_DIR}/run_alphafold.py" \
                --input_dir="$FOLD_INPUT" \
                --model_dir="$WEIGHTS_DIR" \
                --norun_data_pipeline \
                --num_seeds="$NUM_SEEDS" \
                --output_dir="$OUTPUT_DIR" \
                --force_output_dir \
                > "$FOLD_LOG" 2>&1 &
        fi
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
