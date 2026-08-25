#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.
#
# AlphaFast run script -- NO CONTAINER VERSION (v6)
#
# Changes from v5:
#   - Phase 2 builds one global inference queue from all missing jobs.
#   - Each GPU runs one long-lived queue worker and loads AF3 once.
#   - Phase 2 reruns missing jobs via full queue passes, not per-GPU partitions.
#   - Phase 2 only audits after complete queue passes, never mid-pass.
#
# Phase 1 remains job-level resumable and may auto-rerun missing MSA jobs.
#
# Usage:
#   bash /path/to/alphafast/scripts/run_alphafast_local_v6.sh \
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
#       [--template_max_attempts 3] \
#       [--phase2_poll_interval 2.0] \
#       [--phase2_idle_grace_seconds 10.0] \
#       [--jax_compilation_cache_dir /path/to/cache]

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
PHASE2_POLL_INTERVAL="2.0"
PHASE2_IDLE_GRACE_SECONDS="10.0"
JAX_COMPILATION_CACHE_DIR=""
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
    echo "  --input_dir DIR                  Directory containing input JSON files"
    echo "  --output_dir DIR                 Output directory for results"
    echo "  --db_dir DIR                     Genetic database directory"
    echo "  --mmseqs_db_dir DIR              MMseqs2 database directory"
    echo "  --weights_dir DIR                Directory containing af3.bin.zst"
    echo ""
    echo "Optional:"
    echo "  --temp_dir DIR                   Temporary directory for MMseqs work files"
    echo "  --num_gpus N                     Number of GPUs (default: 1)"
    echo "  --batch_size N                   MSA batch size (default: auto)"
    echo "  --gpu_devices IDS                Comma-separated GPU IDs (default: 0 or 0,1,...,N-1)"
    echo "  --num_seeds N                    Number of inference seeds (default: 3)"
    echo "  --mmseqs_threads N               MMseqs2 threads per GPU (default: 15)"
    echo "  --template_batch_size N          Template batch size (default: auto)"
    echo "  --template_max_attempts N        Template retry attempts (default: auto)"
    echo "  --phase2_poll_interval SEC       Queue worker poll interval (default: 2.0)"
    echo "  --phase2_idle_grace_seconds SEC  Queue worker idle grace period (default: 10.0)"
    echo "  --jax_compilation_cache_dir DIR  Optional JAX compilation cache directory"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --db_dir) DB_DIR="$2"; shift 2 ;;
        --mmseqs_db_dir) MMSEQS_DB_DIR="$2"; shift 2 ;;
        --weights_dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --temp_dir) TEMP_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu_devices) GPU_DEVICES="$2"; shift 2 ;;
        --num_seeds) NUM_SEEDS="$2"; shift 2 ;;
        --mmseqs_threads) MMSEQS_THREADS="$2"; shift 2 ;;
        --template_batch_size) TEMPLATE_BATCH_SIZE="$2"; shift 2 ;;
        --template_max_attempts) TEMPLATE_MAX_ATTEMPTS="$2"; shift 2 ;;
        --phase2_poll_interval) PHASE2_POLL_INTERVAL="$2"; shift 2 ;;
        --phase2_idle_grace_seconds) PHASE2_IDLE_GRACE_SECONDS="$2"; shift 2 ;;
        --jax_compilation_cache_dir) JAX_COMPILATION_CACHE_DIR="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required arguments
# ---------------------------------------------------------------------------
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
PHASE2_QUEUE_DIR="${OUTPUT_DIR}/.phase2_queue"

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

for gpu_id in "${GPU_ARRAY[@]}"; do
    mkdir -p "${TEMP_DIR}/gpu_${gpu_id}"
done

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

EMIT_FOLD_QUEUE_SCRIPT="${HELPER_DIR}/emit_fold_queue.py"
cat > "$EMIT_FOLD_QUEUE_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import datetime
import json
import os
import pathlib
import shutil
import sys


resume_base = pathlib.Path(sys.argv[1])
queue_dir = pathlib.Path(sys.argv[2])

if queue_dir.exists():
    shutil.rmtree(queue_dir)

ready_dir = queue_dir / "ready"
in_progress_dir = queue_dir / "in_progress"
done_dir = queue_dir / "done"
failed_dir = queue_dir / "failed"
for path in (ready_dir, in_progress_dir, done_dir, failed_dir):
    path.mkdir(parents=True, exist_ok=True)

seen_names = set()
queued = 0

for resume_dir in sorted(p for p in resume_base.glob("gpu_*") if p.is_dir()):
    for job_entry in sorted(resume_dir.iterdir()):
        name = job_entry.name
        if name in seen_names:
            raise ValueError(f"Duplicate queue entry for job {name}")
        seen_names.add(name)

        target_dir = job_entry.resolve()
        candidates = [
            target_dir / f"{name}_data.json",
            target_dir / "data.json",
        ]
        data_json_path = next((p for p in candidates if p.exists()), None)
        if data_json_path is None:
            raise FileNotFoundError(
                f"No data JSON found for job {name} under {target_dir}"
            )

        token = {
            "name": name,
            "data_json_path": str(data_json_path),
        }
        tmp_path = ready_dir / f".{name}.json.tmp"
        token_path = ready_dir / f"{name}.json"
        with open(tmp_path, "wt") as f:
            json.dump(token, f)
        os.replace(tmp_path, token_path)
        queued += 1

producer_done = {
    "status": "done",
    "total_inputs": queued,
    "timestamp": datetime.datetime.now().isoformat(),
}
tmp_marker = queue_dir / ".producer_done.tmp"
marker_path = queue_dir / "producer_done"
with open(tmp_marker, "wt") as f:
    json.dump(producer_done, f)
os.replace(tmp_marker, marker_path)

print(f"SUMMARY\t{queued}", flush=True)
PYEOF

PHASE2_WORKER_SCRIPT="${HELPER_DIR}/phase2_dynamic_worker.py"
cat > "$PHASE2_WORKER_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
import time
import traceback

DEFAULT_BUCKETS = (
    256, 512, 768, 1024, 1280, 1536, 2048, 2560,
    3072, 3584, 4096, 4608, 5120,
)


def _timestamp() -> str:
    return datetime.datetime.now().isoformat()


def _queue_paths(queue_dir: str) -> dict[str, str]:
    return {
        "ready": os.path.join(queue_dir, "ready"),
        "in_progress": os.path.join(queue_dir, "in_progress"),
        "done": os.path.join(queue_dir, "done"),
        "failed": os.path.join(queue_dir, "failed"),
        "producer_done": os.path.join(queue_dir, "producer_done"),
        "summary": os.path.join(queue_dir, "summary.json"),
        "summary_lock": os.path.join(queue_dir, "summary.lock"),
    }


def _ensure_queue_dirs(queue_dir: str) -> None:
    paths = _queue_paths(queue_dir)
    for key in ("ready", "in_progress", "done", "failed"):
        os.makedirs(paths[key], exist_ok=True)


def _claim_token(queue_dir: str) -> str | None:
    paths = _queue_paths(queue_dir)
    try:
        ready_files = sorted(
            f
            for f in os.listdir(paths["ready"])
            if f.endswith(".json") and not f.startswith(".")
        )
    except FileNotFoundError:
        return None

    for filename in ready_files:
        ready_path = os.path.join(paths["ready"], filename)
        in_progress_path = os.path.join(paths["in_progress"], filename)
        try:
            os.replace(ready_path, in_progress_path)
            return in_progress_path
        except FileNotFoundError:
            continue
    return None


def _read_token(path: str) -> dict[str, str]:
    with open(path, "rt") as f:
        return json.load(f)


def _write_token(path: str, payload: dict[str, object]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wt") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: dict[str, object]) -> None:
    with open(path, "at") as f:
        f.write(json.dumps(payload) + "\n")


def _update_summary(queue_dir: str, record: dict[str, object]) -> None:
    paths = _queue_paths(queue_dir)
    os.makedirs(queue_dir, exist_ok=True)
    with open(paths["summary_lock"], "wt") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(paths["summary"]):
            with open(paths["summary"], "rt") as f:
                summary = json.load(f)
        else:
            summary = {
                "completed": 0,
                "failed": 0,
                "total_inference_seconds": 0.0,
                "last_updated": None,
            }
        status = record.get("status")
        elapsed = float(record.get("elapsed_seconds", 0.0))
        if status == "success":
            summary["completed"] += 1
            summary["total_inference_seconds"] += elapsed
        else:
            summary["failed"] += 1
        if summary["completed"]:
            summary["per_protein_seconds"] = (
                summary["total_inference_seconds"] / summary["completed"]
            )
        summary["last_updated"] = _timestamp()
        _write_token(paths["summary"], summary)


def _maybe_expand_fold_input_seeds(fold_input, num_seeds: int | None):
    if num_seeds is None:
        return fold_input

    current_seed_count = len(fold_input.rng_seeds)
    if num_seeds == 1:
        if current_seed_count == 1:
            print(
                f"[{_timestamp()}] fold job {fold_input.name} already has one"
                " seed; reusing it"
            )
            return fold_input

        first_seed = fold_input.rng_seeds[0]
        print(
            f"[{_timestamp()}] fold job {fold_input.name} already has"
            f" {current_seed_count} seeds; keeping only the first seed"
            f" ({first_seed})"
        )
        return dataclasses.replace(fold_input, rng_seeds=[first_seed])

    if current_seed_count == 1:
        print(
            f"[{_timestamp()}] expanding fold job {fold_input.name} to"
            f" {num_seeds} seeds"
        )
        return fold_input.with_multiple_seeds(num_seeds)

    if current_seed_count == num_seeds:
        print(
            f"[{_timestamp()}] fold job {fold_input.name} already has"
            f" {current_seed_count} seeds; reusing them"
        )
    else:
        print(
            f"[{_timestamp()}] fold job {fold_input.name} already has"
            f" {current_seed_count} seeds; ignoring requested num_seeds="
            f"{num_seeds} and reusing existing seeds"
        )
    return fold_input


def _load_model(
    *,
    model_dir: str,
    gpu_device: int,
    jax_compilation_cache_dir: str | None,
    flash_attention_implementation: str,
    num_diffusion_samples: int,
    num_recycles: int,
):
    import jax

    if jax_compilation_cache_dir is not None:
        jax.config.update(
            "jax_compilation_cache_dir", jax_compilation_cache_dir
        )

    from alphafold3.model.inference import make_model_config
    from alphafold3.model.inference import ModelRunner
    from absl import flags

    # run_alphafold.py normally reaches inference through absl.app.run(main),
    # which parses all registered flags first. The dynamic v6 worker uses
    # argparse, so we parse the global absl flag registry here with defaults to
    # keep tokamax and other absl-backed libraries on the same initialization path.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["phase2_dynamic_worker"])
        print(f"[{_timestamp()}] Parsed absl flags with default values.")

    devices = jax.local_devices(backend="gpu")
    print(
        f"[{_timestamp()}] Found GPUs: {devices}, using device"
        f" {gpu_device}: {devices[gpu_device]}"
    )
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=flash_attention_implementation,
            num_diffusion_samples=num_diffusion_samples,
            num_recycles=num_recycles,
        ),
        device=devices[gpu_device],
        model_dir=pathlib.Path(model_dir),
    )
    print(f"[{_timestamp()}] Loading model parameters...")
    _ = model_runner.model_params
    print(f"[{_timestamp()}] Model loaded successfully.")
    return model_runner


def _run_inference_inprocess(
    *,
    model_runner,
    data_json_path: str,
    output_dir: str,
    buckets: tuple[int, ...] | None,
    num_seeds: int | None,
) -> None:
    from alphafold3.common import folding_input
    from alphafold3.model.inference import process_fold_input

    fold_input = next(
        folding_input.load_fold_inputs_from_path(pathlib.Path(data_json_path))
    )
    fold_input = _maybe_expand_fold_input_seeds(fold_input, num_seeds)
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=None,
        model_runner=model_runner,
        output_dir=output_dir,
        buckets=buckets,
        ref_max_modified_date=datetime.date.fromisoformat("2021-09-30"),
        force_output_dir=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume queued AF3 jobs with one long-lived model per GPU."
    )
    parser.add_argument("--queue_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--idle_grace_seconds", type=float, default=10.0)
    parser.add_argument("--jax_compilation_cache_dir", default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--flash_attention_implementation", default="triton")
    parser.add_argument("--num_diffusion_samples", type=int, default=5)
    parser.add_argument("--num_recycles", type=int, default=10)
    parser.add_argument("--buckets", nargs="*", type=int, default=list(DEFAULT_BUCKETS))
    args = parser.parse_args()

    _ensure_queue_dirs(args.queue_dir)
    timings_path = os.path.join(
        args.queue_dir, f"timings_worker_{args.worker_id}.jsonl"
    )

    print(f"[{_timestamp()}] worker {args.worker_id}: loading model...")
    model_runner = _load_model(
        model_dir=args.model_dir,
        gpu_device=args.gpu_device,
        jax_compilation_cache_dir=args.jax_compilation_cache_dir,
        flash_attention_implementation=args.flash_attention_implementation,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles,
    )
    print(f"[{_timestamp()}] worker {args.worker_id}: model ready")

    buckets = tuple(args.buckets) if args.buckets else None

    idle_start = None
    while True:
        token_path = _claim_token(args.queue_dir)
        if token_path is None:
            producer_done = os.path.exists(
                _queue_paths(args.queue_dir)["producer_done"]
            )
            ready_dir = _queue_paths(args.queue_dir)["ready"]
            ready_empty = True
            try:
                ready_empty = len(os.listdir(ready_dir)) == 0
            except FileNotFoundError:
                ready_empty = True

            if producer_done and ready_empty:
                if idle_start is None:
                    idle_start = time.time()
                elif time.time() - idle_start >= args.idle_grace_seconds:
                    print(f"[{_timestamp()}] worker {args.worker_id}: no work, exiting")
                    break
            else:
                idle_start = None
            time.sleep(args.poll_interval)
            continue

        idle_start = None
        token = _read_token(token_path)
        name = token.get("name")
        data_json_path = token.get("data_json_path")
        if not name or not data_json_path:
            error = "missing name or data_json_path in token"
            print(f"[{_timestamp()}] worker {args.worker_id}: {error}")
            failed_path = os.path.join(
                _queue_paths(args.queue_dir)["failed"], os.path.basename(token_path)
            )
            _write_token(
                failed_path,
                {
                    "status": "failed",
                    "error": error,
                    "worker_id": args.worker_id,
                    "timestamp": _timestamp(),
                },
            )
            continue

        output_dir = os.path.join(args.output_dir, name)
        start_time = time.time()
        print(
            f"[{_timestamp()}] worker {args.worker_id}: running {name} from"
            f" {data_json_path}"
        )
        exit_code = 1
        status = "failed"
        if os.path.exists(data_json_path):
            try:
                _run_inference_inprocess(
                    model_runner=model_runner,
                    data_json_path=data_json_path,
                    output_dir=output_dir,
                    buckets=buckets,
                    num_seeds=args.num_seeds,
                )
                exit_code = 0
                status = "success"
            except Exception as e:
                print(
                    f"[{_timestamp()}] worker {args.worker_id}:"
                    f" inference failed for {name}: {e}"
                )
                print(traceback.format_exc())
                exit_code = 1
                status = "failed"
        else:
            status = "failed"
            exit_code = 2
            print(f"[{_timestamp()}] worker {args.worker_id}: missing {data_json_path}")

        elapsed = time.time() - start_time
        record = {
            "name": name,
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 3),
            "worker_id": args.worker_id,
            "timestamp": _timestamp(),
        }
        _append_jsonl(timings_path, record)
        _update_summary(args.queue_dir, record)

        dest_dir = _queue_paths(args.queue_dir)[
            "done" if status == "success" else "failed"
        ]
        dest_path = os.path.join(dest_dir, os.path.basename(token_path))
        _write_token(dest_path, {**token, **record})
        try:
            os.remove(token_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
PYEOF

chmod +x "$PARTITION_SCRIPT" "$MSA_RESUME_SCRIPT" "$FOLD_RESUME_SCRIPT" "$EMIT_FOLD_QUEUE_SCRIPT" "$PHASE2_WORKER_SCRIPT"

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
PHASE2_QUEUED=0
PHASE2_WORKER_FAILURES=0

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

parse_queue_summary_line() {
    local summary_line="$1"
    local tag
    IFS=$'\t' read -r tag PHASE2_QUEUED <<< "$summary_line"
    if [ "$tag" != "SUMMARY" ]; then
        echo "ERROR: Unexpected queue helper output: $summary_line"
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
# Phase 1 runners
# ---------------------------------------------------------------------------
run_single_msa_phase() {
    local pass_index=1
    local log_path
    local current_batch
    local local_status

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

# ---------------------------------------------------------------------------
# Phase 2 runner: global dynamic queue
# ---------------------------------------------------------------------------
run_dynamic_fold_phase() {
    local queue_summary_line
    local max_workers
    local idx
    local gpu_id
    local worker_id
    local log_path
    local status
    local -a worker_pids=()
    local -a worker_logs=()
    local -a worker_cmd=()

    queue_summary_line=$(python3 "$EMIT_FOLD_QUEUE_SCRIPT" "$FOLD_RESUME_BASE" "$PHASE2_QUEUE_DIR")
    parse_queue_summary_line "$queue_summary_line"

    if [ "$PHASE2_QUEUED" -ne "$FOLD_TOTAL_TODO" ]; then
        echo "ERROR: Phase 2 queue contains ${PHASE2_QUEUED} job(s) but audit reported ${FOLD_TOTAL_TODO} missing job(s)."
        exit 1
    fi

    if [ "$PHASE2_QUEUED" -eq 0 ]; then
        echo "ERROR: Phase 2 queue is empty but missing outputs remain."
        exit 1
    fi

    max_workers="$NUM_GPUS"
    if [ "$PHASE2_QUEUED" -lt "$max_workers" ]; then
        max_workers="$PHASE2_QUEUED"
    fi

    echo "--- Phase 2: Dynamic Fold Queue ---"
    echo "Queued missing jobs: ${PHASE2_QUEUED}/${FOLD_TOTAL_EXPECTED}"
    echo "Queue dir: ${PHASE2_QUEUE_DIR}"
    echo "Workers: ${max_workers}"

    worker_pids=()
    worker_logs=()
    PHASE2_WORKER_FAILURES=0

    for (( idx=0; idx<max_workers; idx++ )); do
        gpu_id="${GPU_ARRAY[$idx]}"
        worker_id="gpu${gpu_id}"
        log_path="${LOG_DIR}/fold_worker_${worker_id}_${TIMESTAMP}.log"

        worker_cmd=(
            python -u "$PHASE2_WORKER_SCRIPT"
            --queue_dir="$PHASE2_QUEUE_DIR"
            --output_dir="$OUTPUT_DIR"
            --worker_id="$worker_id"
            --model_dir="$WEIGHTS_DIR"
            --gpu_device=0
            --poll_interval="$PHASE2_POLL_INTERVAL"
            --idle_grace_seconds="$PHASE2_IDLE_GRACE_SECONDS"
            --num_seeds="$NUM_SEEDS"
        )
        if [ -n "$JAX_COMPILATION_CACHE_DIR" ]; then
            worker_cmd+=("--jax_compilation_cache_dir=$JAX_COMPILATION_CACHE_DIR")
        fi

        echo "  GPU ${gpu_id}: worker=${worker_id} log=${log_path}"
        CUDA_VISIBLE_DEVICES="$gpu_id" "${worker_cmd[@]}" > "$log_path" 2>&1 &
        worker_pids+=("$!")
        worker_logs+=("$log_path")
    done

    echo ""
    echo "Waiting for all fold workers to finish..."
    set +e
    for idx in "${!worker_pids[@]}"; do
        status=0
        wait "${worker_pids[$idx]}"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "WARN: Fold worker PID ${worker_pids[$idx]} exited with status ${status} (log: ${worker_logs[$idx]})"
            PHASE2_WORKER_FAILURES=$((PHASE2_WORKER_FAILURES + 1))
        fi
    done
    set -e
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=========================================="
echo "AlphaFast Run (Native, No Container) v6"
echo "=========================================="
echo "App dir:                    $APP_DIR"
echo "Input dir:                  $INPUT_DIR"
echo "Output dir:                 $OUTPUT_DIR"
echo "DB dir:                     $DB_DIR"
echo "MMseqs dir:                 $MMSEQS_DB_DIR"
echo "Weights:                    $WEIGHTS_DIR"
echo "Temp dir:                   $TEMP_DIR"
echo "GPUs:                       $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size:                 $BATCH_SIZE"
if [ -n "$AUTO_BATCH_SIZE_MODE" ]; then
    echo "Batch mode:                 $AUTO_BATCH_SIZE_MODE"
fi
echo "Num seeds:                  $NUM_SEEDS"
echo "MMseqs threads:             $MMSEQS_THREADS per GPU"
echo "Phase 1 auto reruns:        $MAX_AUTO_RERUNS"
echo "Phase 2 worker poll:        $PHASE2_POLL_INTERVAL s"
echo "Phase 2 idle grace:         $PHASE2_IDLE_GRACE_SECONDS s"
if [ -n "$JAX_COMPILATION_CACHE_DIR" ]; then
    echo "JAX compilation cache dir:  $JAX_COMPILATION_CACHE_DIR"
fi
echo "Log dir:                    $LOG_DIR"
echo "Start time:                 $(date)"
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
# Phase 2: pre-check, dynamic queue passes, end-of-pass audits only
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
    phase2_pass_index=1
    if [ -f "$PHASE2_DONE" ]; then
        echo "Ignoring stale phase 2 marker: $PHASE2_DONE"
        rm -f "$PHASE2_DONE"
    fi
    echo ""

    while [ "$FOLD_TOTAL_TODO" -gt 0 ]; do
        if [ "$phase2_pass_index" -gt "$MAX_STAGE_PASSES" ]; then
            echo "ERROR: Phase 2 still has ${FOLD_TOTAL_TODO}/${FOLD_TOTAL_EXPECTED} missing job(s) after ${MAX_AUTO_RERUNS} auto-reruns."
            echo "Re-run v6 to resume only the remaining missing jobs."
            exit 1
        fi

        echo "=== Phase 2 queue pass $(pass_label "$phase2_pass_index") ==="
        run_dynamic_fold_phase

        if [ "$NUM_GPUS" -gt 1 ]; then
            prepare_multi_fold_resume
        else
            prepare_single_fold_resume
        fi

        echo "Phase 2 audit after pass ${phase2_pass_index}: done=${FOLD_TOTAL_DONE}, missing=${FOLD_TOTAL_TODO}, expected=${FOLD_TOTAL_EXPECTED}"

        if [ "$PHASE2_WORKER_FAILURES" -ne 0 ]; then
            echo "WARN: ${PHASE2_WORKER_FAILURES} phase-2 worker(s) exited non-zero in pass ${phase2_pass_index}. Missing jobs will be retried if any remain."
        fi
        echo ""

        phase2_pass_index=$((phase2_pass_index + 1))
    done

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
