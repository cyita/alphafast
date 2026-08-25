#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.
#
# AlphaFast run script -- NO CONTAINER VERSION (v5)
#
# Changes from v2:
#   - Phase 1 supports job-level resume by auditing per-job MSA outputs.
#   - Phase 1 auto-reruns missing jobs up to 3 times after the initial pass.
#   - Phase 2 supports job-level resume by auditing final inference outputs.
#   - Phase 2 auto-reruns missing jobs up to 3 times after the initial pass.
#   - .phase1_done and .phase2_done are advisory only; stale markers are ignored.
#
# Usage:
#   bash /path/to/alphafast/scripts/run_alphafast_local_v5.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir /path/to/databases \
#       --mmseqs_db_dir /path/to/mmseqs \
#       --weights_dir /path/to/weights \
#       [--temp_dir /path/to/temp] \
#       [--num_gpus 1] \
#       [--batch_size 500] \
#       [--gpu_devices 0,1,...] \
#       [--num_seeds 3] \
#       [--mmseqs_threads 15] \
#       [--template_batch_size 128] \
#       [--template_max_attempts 3]

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
MMSEQS_THREADS=15
TEMPLATE_BATCH_SIZE=""
TEMPLATE_MAX_ATTEMPTS=""
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MAX_AUTO_RERUNS=3
MAX_STAGE_PASSES=$((MAX_AUTO_RERUNS + 1))

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir DIR --output_dir DIR --db_dir DIR --mmseqs_db_dir DIR --weights_dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input_dir DIR            Directory containing input JSON files"
    echo "  --output_dir DIR           Output directory for results"
    echo "  --db_dir DIR               Genetic database directory"
    echo "  --mmseqs_db_dir DIR        MMseqs2 database directory"
    echo "  --weights_dir DIR          Directory containing af3.bin.zst"
    echo ""
    echo "Optional:"
    echo "  --temp_dir DIR             Temporary directory for MMseqs work files"
    echo "  --num_gpus N               Number of GPUs (default: 1)"
    echo "  --batch_size N             MSA batch size (default: auto)"
    echo "  --gpu_devices IDS          Comma-separated GPU IDs (default: 0 or 0,1,...,N-1)"
    echo "  --num_seeds N              Number of inference seeds (default: 3)"
    echo "  --mmseqs_threads N         MMseqs2 threads per GPU (default: 15)"
    echo "  --template_batch_size N    Template batch size (default: auto)"
    echo "  --template_max_attempts N  Template retry attempts (default: auto)"
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

for d in "$INPUT_DIR" "$DB_DIR" "$MMSEQS_DB_DIR" "$WEIGHTS_DIR" "$APP_DIR"; do
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
MMSEQS_DB_DIR="$(cd "$MMSEQS_DB_DIR" && pwd)"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PHASE1_DONE="${OUTPUT_DIR}/.phase1_done"
PHASE2_DONE="${OUTPUT_DIR}/.phase2_done"

if [ -z "$TEMP_DIR" ]; then
    TEMP_DIR="${OUTPUT_DIR}/temp_${TIMESTAMP}"
fi
mkdir -p "$TEMP_DIR"
TEMP_DIR="$(cd "$TEMP_DIR" && pwd)"

HELPER_DIR="${TEMP_DIR}/helpers"
mkdir -p "$HELPER_DIR"

MSA_RESUME_BASE="${OUTPUT_DIR}/.msa_resume"
FOLD_RESUME_BASE="${OUTPUT_DIR}/.fold_resume"
MSA_OUTPUT_DIR="${OUTPUT_DIR}/msa_output"

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

if [ "${#GPU_ARRAY[@]}" -lt "$NUM_GPUS" ]; then
    echo "ERROR: --gpu_devices provides ${#GPU_ARRAY[@]} GPU IDs but --num_gpus=${NUM_GPUS}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper scripts
# ---------------------------------------------------------------------------
PARTITION_SCRIPT="${HELPER_DIR}/partition_all.py"
cat > "$PARTITION_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json
import pathlib
import shutil
import sys


def load_single_job(json_path: pathlib.Path) -> dict:
    raw = json.loads(json_path.read_text())
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"{json_path} contains {len(raw)} jobs; partitioning expects one job per file."
            )
        raw = raw[0]
    if "name" not in raw:
        raise ValueError(f"{json_path} does not contain a 'name' field.")
    return raw


input_dir = pathlib.Path(sys.argv[1])
num_gpus = int(sys.argv[2])
part_prefix = sys.argv[3]

json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "index.json")
if not json_files:
    raise SystemExit(f"No .json files found in {input_dir}")

part_dirs = []
for idx in range(num_gpus):
    part_dir = pathlib.Path(f"{part_prefix}{idx}")
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_dirs.append(part_dir)

for idx, src in enumerate(json_files):
    dst_dir = part_dirs[idx % num_gpus]
    data = load_single_job(src)
    json_dir = src.parent.resolve()

    for seq in data.get("sequences", []):
        for chain_type in ("protein", "rna", "dna"):
            chain = seq.get(chain_type)
            if not chain:
                continue
            for key in ("unpairedMsaPath", "pairedMsaPath", "templatesPath"):
                value = chain.get(key)
                if value and not pathlib.Path(value).is_absolute():
                    chain[key] = str((json_dir / value).resolve())

    (dst_dir / src.name).write_text(json.dumps(data, indent=2))

print(f"Partitioned {len(json_files)} input files across {num_gpus} partition(s).", flush=True)
PYEOF

MSA_RESUME_SCRIPT="${HELPER_DIR}/prepare_msa_resume.py"
cat > "$MSA_RESUME_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json
import pathlib
import shutil
import string
import sys


def sanitise(name: str) -> str:
    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.")
    return "".join(ch for ch in spaceless_name if ch in allowed_chars)


def load_job_name(json_path: pathlib.Path) -> str:
    raw = json.loads(json_path.read_text())
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"{json_path} contains {len(raw)} jobs; resume mode expects one job per file."
            )
        raw = raw[0]
    return sanitise(raw["name"])


expected_dir = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
resume_dir = pathlib.Path(sys.argv[3])

if resume_dir.exists():
    shutil.rmtree(resume_dir)
resume_dir.mkdir(parents=True, exist_ok=True)

done = 0
todo = 0
total = 0

for src in sorted(p for p in expected_dir.glob("*.json") if p.name != "index.json"):
    total += 1
    job_name = load_job_name(src)
    job_dir = output_dir / job_name
    complete = (
        (job_dir / "data.json").exists()
        or (job_dir / f"{job_name}_data.json").exists()
    )
    if complete:
        done += 1
        continue
    (resume_dir / src.name).symlink_to(src.resolve())
    todo += 1

print(f"SUMMARY\t{done}\t{todo}\t{total}", flush=True)
PYEOF

FOLD_RESUME_SCRIPT="${HELPER_DIR}/prepare_fold_resume.py"
cat > "$FOLD_RESUME_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json
import pathlib
import shutil
import string
import sys


def sanitise(name: str) -> str:
    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.")
    return "".join(ch for ch in spaceless_name if ch in allowed_chars)


def load_job_name(json_path: pathlib.Path) -> str:
    raw = json.loads(json_path.read_text())
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"{json_path} contains {len(raw)} jobs; resume mode expects one job per file."
            )
        raw = raw[0]
    return sanitise(raw["name"])


expected_dir = pathlib.Path(sys.argv[1])
msa_dir = pathlib.Path(sys.argv[2])
final_output_dir = pathlib.Path(sys.argv[3])
resume_dir = pathlib.Path(sys.argv[4])

if resume_dir.exists():
    shutil.rmtree(resume_dir)
resume_dir.mkdir(parents=True, exist_ok=True)

done = 0
todo = 0
total = 0

for src in sorted(p for p in expected_dir.glob("*.json") if p.name != "index.json"):
    total += 1
    job_name = load_job_name(src)
    job_dir = msa_dir / job_name
    has_msa = (
        (job_dir / "data.json").exists()
        or (job_dir / f"{job_name}_data.json").exists()
    )
    if not has_msa:
        continue

    output_dir = final_output_dir / job_name
    complete = (
        (output_dir / f"{job_name}_model.cif").exists()
        and (output_dir / f"{job_name}_ranking_scores.csv").exists()
    )
    if complete:
        done += 1
        continue

    (resume_dir / job_name).symlink_to(job_dir.resolve())
    todo += 1

print(f"SUMMARY\t{done}\t{todo}\t{total}", flush=True)
PYEOF

chmod +x "$PARTITION_SCRIPT" "$MSA_RESUME_SCRIPT" "$FOLD_RESUME_SCRIPT"

# ---------------------------------------------------------------------------
# Summary parsing and batch helpers
# ---------------------------------------------------------------------------
SUMMARY_DONE=0
SUMMARY_TODO=0
SUMMARY_TOTAL=0

MSA_TOTAL_DONE=0
MSA_TOTAL_TODO=0
MSA_TOTAL_EXPECTED=0
FOLD_TOTAL_DONE=0
FOLD_TOTAL_TODO=0
FOLD_TOTAL_EXPECTED=0

declare -a MSA_TODO_COUNTS=()
declare -a MSA_EXPECTED_COUNTS=()
declare -a FOLD_TODO_COUNTS=()
declare -a FOLD_EXPECTED_COUNTS=()

parse_summary_line() {
    local summary_line="$1"
    local tag
    IFS=$'\t' read -r tag SUMMARY_DONE SUMMARY_TODO SUMMARY_TOTAL <<< "$summary_line"
    if [ "$tag" != "SUMMARY" ]; then
        echo "ERROR: Unexpected helper output: $summary_line"
        exit 1
    fi
}

calc_reduced_batch_size() {
    local base_batch="$1"
    local pass_index="$2"
    local todo_count="$3"
    local batch="$base_batch"
    local idx

    if [ "$batch" -lt 1 ]; then
        batch=1
    fi

    for (( idx=1; idx<pass_index; idx++ )); do
        batch=$(( (batch + 1) / 2 ))
        if [ "$batch" -lt 1 ]; then
            batch=1
        fi
    done

    if [ "$todo_count" -lt "$batch" ]; then
        batch="$todo_count"
    fi
    if [ "$batch" -lt 1 ]; then
        batch=1
    fi
    echo "$batch"
}

pass_label() {
    local pass_index="$1"
    if [ "$pass_index" -eq 1 ]; then
        echo "initial pass"
    else
        echo "auto-rerun $((pass_index - 1))/${MAX_AUTO_RERUNS}"
    fi
}

# ---------------------------------------------------------------------------
# Resume preparation helpers
# ---------------------------------------------------------------------------
prepare_single_msa_resume() {
    local summary_line
    summary_line=$(python3 "$MSA_RESUME_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" "${MSA_RESUME_BASE}/gpu_0")
    parse_summary_line "$summary_line"

    MSA_TODO_COUNTS=("$SUMMARY_TODO")
    MSA_EXPECTED_COUNTS=("$SUMMARY_TOTAL")
    MSA_TOTAL_DONE="$SUMMARY_DONE"
    MSA_TOTAL_TODO="$SUMMARY_TODO"
    MSA_TOTAL_EXPECTED="$SUMMARY_TOTAL"
}

prepare_multi_msa_resume() {
    local total_done=0
    local total_todo=0
    local total_expected=0
    local summary_line
    local part_input
    local part_output
    local resume_dir
    local idx

    MSA_TODO_COUNTS=()
    MSA_EXPECTED_COUNTS=()

    for (( idx=0; idx<NUM_GPUS; idx++ )); do
        part_input="${OUTPUT_DIR}/partition_${idx}"
        part_output="${MSA_OUTPUT_DIR}/gpu_${idx}"
        resume_dir="${MSA_RESUME_BASE}/gpu_${idx}"

        summary_line=$(python3 "$MSA_RESUME_SCRIPT" "$part_input" "$part_output" "$resume_dir")
        parse_summary_line "$summary_line"

        MSA_TODO_COUNTS[$idx]="$SUMMARY_TODO"
        MSA_EXPECTED_COUNTS[$idx]="$SUMMARY_TOTAL"
        total_done=$((total_done + SUMMARY_DONE))
        total_todo=$((total_todo + SUMMARY_TODO))
        total_expected=$((total_expected + SUMMARY_TOTAL))
    done

    MSA_TOTAL_DONE="$total_done"
    MSA_TOTAL_TODO="$total_todo"
    MSA_TOTAL_EXPECTED="$total_expected"
}

prepare_single_fold_resume() {
    local summary_line
    summary_line=$(python3 "$FOLD_RESUME_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" "${FOLD_RESUME_BASE}/gpu_0")
    parse_summary_line "$summary_line"

    FOLD_TODO_COUNTS=("$SUMMARY_TODO")
    FOLD_EXPECTED_COUNTS=("$SUMMARY_TOTAL")
    FOLD_TOTAL_DONE="$SUMMARY_DONE"
    FOLD_TOTAL_TODO="$SUMMARY_TODO"
    FOLD_TOTAL_EXPECTED="$SUMMARY_TOTAL"
}

prepare_multi_fold_resume() {
    local total_done=0
    local total_todo=0
    local total_expected=0
    local summary_line
    local expected_dir
    local msa_dir
    local resume_dir
    local idx

    FOLD_TODO_COUNTS=()
    FOLD_EXPECTED_COUNTS=()

    for (( idx=0; idx<NUM_GPUS; idx++ )); do
        expected_dir="${OUTPUT_DIR}/partition_${idx}"
        msa_dir="${MSA_OUTPUT_DIR}/gpu_${idx}"
        resume_dir="${FOLD_RESUME_BASE}/gpu_${idx}"

        summary_line=$(python3 "$FOLD_RESUME_SCRIPT" "$expected_dir" "$msa_dir" "$OUTPUT_DIR" "$resume_dir")
        parse_summary_line "$summary_line"

        FOLD_TODO_COUNTS[$idx]="$SUMMARY_TODO"
        FOLD_EXPECTED_COUNTS[$idx]="$SUMMARY_TOTAL"
        total_done=$((total_done + SUMMARY_DONE))
        total_todo=$((total_todo + SUMMARY_TODO))
        total_expected=$((total_expected + SUMMARY_TOTAL))
    done

    FOLD_TOTAL_DONE="$total_done"
    FOLD_TOTAL_TODO="$total_todo"
    FOLD_TOTAL_EXPECTED="$total_expected"
}

# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
run_single_msa_phase() {
    local pass_index=1
    local log_path
    local current_batch

    while [ "$MSA_TOTAL_TODO" -gt 0 ]; do
        if [ "$pass_index" -gt "$MAX_STAGE_PASSES" ]; then
            echo "ERROR: Phase 1 still has ${MSA_TOTAL_TODO}/${MSA_TOTAL_EXPECTED} missing job(s) after ${MAX_AUTO_RERUNS} auto-reruns."
            exit 1
        fi

        current_batch=$(calc_reduced_batch_size "$BATCH_SIZE" "$pass_index" "$MSA_TOTAL_TODO")
        log_path="${LOG_DIR}/pipeline_pass${pass_index}_${TIMESTAMP}.log"

        echo "=== Stage 1: Data Pipeline ($(pass_label "$pass_index")) ==="
        echo "GPU: ${GPU_ARRAY[0]} | Missing: ${MSA_TOTAL_TODO}/${MSA_TOTAL_EXPECTED} | Batch size: ${current_batch} | Log: ${log_path}"
        echo ""

        set +e
        CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" python "${APP_DIR}/run_data_pipeline.py" \
            --input_dir="${MSA_RESUME_BASE}/gpu_0" \
            --output_dir="$OUTPUT_DIR" \
            --db_dir="$DB_DIR" \
            --mmseqs_db_dir="$MMSEQS_DB_DIR" \
            --use_mmseqs_gpu \
            --batch_size="$current_batch" \
            --mmseqs_n_threads="$MMSEQS_THREADS" \
            --temp_dir="${TEMP_DIR}/gpu_${GPU_ARRAY[0]}" \
            ${TEMPLATE_BATCH_SIZE:+--template_batch_size="$TEMPLATE_BATCH_SIZE"} \
            ${TEMPLATE_MAX_ATTEMPTS:+--template_max_attempts="$TEMPLATE_MAX_ATTEMPTS"} \
            2>&1 | tee "$log_path"
        local_status=$?
        set -e

        if [ "$local_status" -ne 0 ]; then
            echo "WARN: Stage 1 pass ${pass_index} exited with status ${local_status}. Auditing outputs before deciding whether to continue."
        fi

        prepare_single_msa_resume
        echo "Stage 1 audit after pass ${pass_index}: done=${MSA_TOTAL_DONE}, missing=${MSA_TOTAL_TODO}, expected=${MSA_TOTAL_EXPECTED}"
        echo ""

        pass_index=$((pass_index + 1))
    done
}

run_multi_msa_phase() {
    local pass_index=1
    local idx
    local part_input
    local base_batch
    local current_batch
    local log_path
    local gpu_id
    local resume_dir
    local todo_count
    local launched=0
    local failed=0
    local -a msa_pids=()

    while [ "$MSA_TOTAL_TODO" -gt 0 ]; do
        if [ "$pass_index" -gt "$MAX_STAGE_PASSES" ]; then
            echo "ERROR: Phase 1 still has ${MSA_TOTAL_TODO}/${MSA_TOTAL_EXPECTED} missing job(s) after ${MAX_AUTO_RERUNS} auto-reruns."
            exit 1
        fi

        echo "--- Phase 1: Parallel MSA ($(pass_label "$pass_index")) ---"
        echo "Missing jobs before pass ${pass_index}: ${MSA_TOTAL_TODO}/${MSA_TOTAL_EXPECTED}"

        msa_pids=()
        launched=0

        for (( idx=0; idx<NUM_GPUS; idx++ )); do
            todo_count="${MSA_TODO_COUNTS[$idx]:-0}"
            if [ "$todo_count" -eq 0 ]; then
                echo "  GPU ${GPU_ARRAY[$idx]}: no missing MSA jobs, skipping."
                continue
            fi

            part_input="${OUTPUT_DIR}/partition_${idx}"
            base_batch=$(find "$part_input" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
            if [ "$base_batch" -gt "$BATCH_SIZE" ]; then
                base_batch="$BATCH_SIZE"
            fi
            current_batch=$(calc_reduced_batch_size "$base_batch" "$pass_index" "$todo_count")

            gpu_id="${GPU_ARRAY[$idx]}"
            resume_dir="${MSA_RESUME_BASE}/gpu_${idx}"
            log_path="${LOG_DIR}/msa_gpu${gpu_id}_pass${pass_index}_${TIMESTAMP}.log"

            echo "  GPU ${gpu_id}: missing=${todo_count}/${MSA_EXPECTED_COUNTS[$idx]} batch_size=${current_batch} log=${log_path}"
            (
                CUDA_VISIBLE_DEVICES="$gpu_id" python "${APP_DIR}/run_data_pipeline.py" \
                    --input_dir="$resume_dir" \
                    --output_dir="${MSA_OUTPUT_DIR}/gpu_${idx}" \
                    --db_dir="$DB_DIR" \
                    --mmseqs_db_dir="$MMSEQS_DB_DIR" \
                    --use_mmseqs_gpu \
                    --batch_size="$current_batch" \
                    --mmseqs_n_threads="$MMSEQS_THREADS" \
                    --temp_dir="${TEMP_DIR}/gpu_${gpu_id}" \
                    ${TEMPLATE_BATCH_SIZE:+--template_batch_size="$TEMPLATE_BATCH_SIZE"} \
                    ${TEMPLATE_MAX_ATTEMPTS:+--template_max_attempts="$TEMPLATE_MAX_ATTEMPTS"} \
                    >> "$log_path" 2>&1
            ) &
            msa_pids+=("$!")
            launched=$((launched + 1))
        done

        if [ "$launched" -eq 0 ]; then
            break
        fi

        echo ""
        echo "Waiting for all MSA jobs to finish..."
        set +e
        failed=0
        for pid in "${msa_pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        set -e

        if [ "$failed" -ne 0 ]; then
            echo "WARN: One or more MSA jobs failed in pass ${pass_index}. Auditing outputs before deciding whether to continue."
        fi

        prepare_multi_msa_resume
        echo "Stage 1 audit after pass ${pass_index}: done=${MSA_TOTAL_DONE}, missing=${MSA_TOTAL_TODO}, expected=${MSA_TOTAL_EXPECTED}"
        echo ""

        pass_index=$((pass_index + 1))
    done
}

run_single_fold_phase() {
    local pass_index=1
    local log_path
    local local_status

    while [ "$FOLD_TOTAL_TODO" -gt 0 ]; do
        if [ "$pass_index" -gt "$MAX_STAGE_PASSES" ]; then
            echo "ERROR: Phase 2 still has ${FOLD_TOTAL_TODO}/${FOLD_TOTAL_EXPECTED} missing job(s) after ${MAX_AUTO_RERUNS} auto-reruns."
            exit 1
        fi

        log_path="${LOG_DIR}/inference_pass${pass_index}_${TIMESTAMP}.log"

        echo "=== Stage 2: Inference ($(pass_label "$pass_index")) ==="
        echo "GPU: ${GPU_ARRAY[0]} | Missing: ${FOLD_TOTAL_TODO}/${FOLD_TOTAL_EXPECTED} | Log: ${log_path}"
        echo ""

        set +e
        CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" python "${APP_DIR}/run_alphafold.py" \
            --input_dir="${FOLD_RESUME_BASE}/gpu_0" \
            --model_dir="$WEIGHTS_DIR" \
            --norun_data_pipeline \
            --num_seeds="$NUM_SEEDS" \
            --output_dir="$OUTPUT_DIR" \
            --force_output_dir \
            2>&1 | tee "$log_path"
        local_status=$?
        set -e

        if [ "$local_status" -ne 0 ]; then
            echo "WARN: Stage 2 pass ${pass_index} exited with status ${local_status}. Auditing outputs before deciding whether to continue."
        fi

        prepare_single_fold_resume
        echo "Stage 2 audit after pass ${pass_index}: done=${FOLD_TOTAL_DONE}, missing=${FOLD_TOTAL_TODO}, expected=${FOLD_TOTAL_EXPECTED}"
        echo ""

        pass_index=$((pass_index + 1))
    done
}

run_multi_fold_phase() {
    local pass_index=1
    local idx
    local gpu_id
    local resume_dir
    local log_path
    local todo_count
    local launched=0
    local failed=0
    local -a fold_pids=()

    while [ "$FOLD_TOTAL_TODO" -gt 0 ]; do
        if [ "$pass_index" -gt "$MAX_STAGE_PASSES" ]; then
            echo "ERROR: Phase 2 still has ${FOLD_TOTAL_TODO}/${FOLD_TOTAL_EXPECTED} missing job(s) after ${MAX_AUTO_RERUNS} auto-reruns."
            exit 1
        fi

        echo "--- Phase 2: Parallel Fold ($(pass_label "$pass_index")) ---"
        echo "Missing jobs before pass ${pass_index}: ${FOLD_TOTAL_TODO}/${FOLD_TOTAL_EXPECTED}"

        fold_pids=()
        launched=0

        for (( idx=0; idx<NUM_GPUS; idx++ )); do
            todo_count="${FOLD_TODO_COUNTS[$idx]:-0}"
            if [ "$todo_count" -eq 0 ]; then
                echo "  GPU ${GPU_ARRAY[$idx]}: no missing fold jobs, skipping."
                continue
            fi

            gpu_id="${GPU_ARRAY[$idx]}"
            resume_dir="${FOLD_RESUME_BASE}/gpu_${idx}"
            log_path="${LOG_DIR}/fold_gpu${gpu_id}_pass${pass_index}_${TIMESTAMP}.log"

            echo "  GPU ${gpu_id}: missing=${todo_count}/${FOLD_EXPECTED_COUNTS[$idx]} log=${log_path}"
            CUDA_VISIBLE_DEVICES="$gpu_id" python "${APP_DIR}/run_alphafold.py" \
                --input_dir="$resume_dir" \
                --model_dir="$WEIGHTS_DIR" \
                --norun_data_pipeline \
                --num_seeds="$NUM_SEEDS" \
                --output_dir="$OUTPUT_DIR" \
                --force_output_dir \
                > "$log_path" 2>&1 &
            fold_pids+=("$!")
            launched=$((launched + 1))
        done

        if [ "$launched" -eq 0 ]; then
            break
        fi

        echo ""
        echo "Waiting for all fold jobs to finish..."
        set +e
        failed=0
        for pid in "${fold_pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        set -e

        if [ "$failed" -ne 0 ]; then
            echo "WARN: One or more fold jobs failed in pass ${pass_index}. Auditing outputs before deciding whether to continue."
        fi

        prepare_multi_fold_resume
        echo "Stage 2 audit after pass ${pass_index}: done=${FOLD_TOTAL_DONE}, missing=${FOLD_TOTAL_TODO}, expected=${FOLD_TOTAL_EXPECTED}"
        echo ""

        pass_index=$((pass_index + 1))
    done
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=========================================="
echo "AlphaFast Run (Native, No Container) v5"
echo "=========================================="
echo "App dir:        $APP_DIR"
echo "Input dir:      $INPUT_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "DB dir:         $DB_DIR"
echo "MMseqs dir:     $MMSEQS_DB_DIR"
echo "Weights:        $WEIGHTS_DIR"
echo "Temp dir:       $TEMP_DIR"
echo "GPUs:           $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size:     $BATCH_SIZE"
if [ -n "$AUTO_BATCH_SIZE_MODE" ]; then
    echo "Batch mode:     $AUTO_BATCH_SIZE_MODE"
fi
echo "Num seeds:      $NUM_SEEDS"
echo "MMseqs threads: $MMSEQS_THREADS per GPU"
echo "Auto reruns:    $MAX_AUTO_RERUNS per stage"
echo "Log dir:        $LOG_DIR"
echo "Start time:     $(date)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Phase 1: audit and resume
# ---------------------------------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    mkdir -p "$MSA_OUTPUT_DIR"
    echo "--- Partitioning inputs round-robin across $NUM_GPUS GPUs ---"
    python3 "$PARTITION_SCRIPT" "$INPUT_DIR" "$NUM_GPUS" "${OUTPUT_DIR}/partition_"
    echo ""
    prepare_multi_msa_resume
else
    prepare_single_msa_resume
fi

echo "Phase 1 pre-check: done=${MSA_TOTAL_DONE}, missing=${MSA_TOTAL_TODO}, expected=${MSA_TOTAL_EXPECTED}"

if [ "$MSA_TOTAL_EXPECTED" -eq 0 ]; then
    echo "ERROR: Phase 1 audit found zero expected inputs."
    exit 1
fi

if [ "$MSA_TOTAL_TODO" -eq 0 ]; then
    touch "$PHASE1_DONE"
    echo "Phase 1 already complete; audit passed."
    echo ""
else
    if [ -f "$PHASE1_DONE" ]; then
        echo "Ignoring stale phase 1 marker: $PHASE1_DONE"
        rm -f "$PHASE1_DONE"
    fi
    if [ -f "$PHASE2_DONE" ]; then
        echo "Removing phase 2 marker because phase 1 is incomplete: $PHASE2_DONE"
        rm -f "$PHASE2_DONE"
    fi
    echo ""

    if [ "$NUM_GPUS" -gt 1 ]; then
        run_multi_msa_phase
    else
        run_single_msa_phase
    fi

    touch "$PHASE1_DONE"
    echo "Phase 1 complete. Marker written: $PHASE1_DONE"
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 2: audit and resume
# ---------------------------------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    prepare_multi_fold_resume
else
    prepare_single_fold_resume
fi

echo "Phase 2 pre-check: done=${FOLD_TOTAL_DONE}, missing=${FOLD_TOTAL_TODO}, expected=${FOLD_TOTAL_EXPECTED}"

if [ "$FOLD_TOTAL_EXPECTED" -eq 0 ]; then
    echo "ERROR: Phase 2 audit found zero expected inputs."
    exit 1
fi

if [ "$FOLD_TOTAL_TODO" -eq 0 ]; then
    touch "$PHASE2_DONE"
    echo "Phase 2 already complete; audit passed."
else
    if [ -f "$PHASE2_DONE" ]; then
        echo "Ignoring stale phase 2 marker: $PHASE2_DONE"
        rm -f "$PHASE2_DONE"
    fi
    echo ""

    if [ "$NUM_GPUS" -gt 1 ]; then
        run_multi_fold_phase
    else
        run_single_fold_phase
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
