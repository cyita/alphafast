#!/usr/bin/env python3
# Copyright 2026 Romero Lab, Duke University
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast.

"""Rerun failed MSA partitions without repartitioning.

This script is intended for phase-1 recovery when some logical partitions
failed, but the successful partitions under msa_output/gpu_X should be kept.

Each recovery unit is a manual pair:
  --partition_pair <partition_input_dir> <msa_output_dir>

Typical usage for one failed task:
  python scripts/run_msa_partition_recovery.py \
      --db_dir /path/to/databases \
      --mmseqs_db_dir /path/to/mmseqs \
      --gpu_devices 0,2 \
      --partition_pair /taskA/output/partition_1 /taskA/output/msa_output/gpu_1 \
      --partition_pair /taskA/output/partition_3 /taskA/output/msa_output/gpu_3

You can also mix failed partitions from different tasks in one run:
  python scripts/run_msa_partition_recovery.py \
      --db_dir /path/to/databases \
      --mmseqs_db_dir /path/to/mmseqs \
      --gpu_devices 0,1,2,3 \
      --partition_pair /taskA/output/partition_1 /taskA/output/msa_output/gpu_1 \
      --partition_pair /taskB/output/partition_0 /taskB/output/msa_output/gpu_0 \
      --partition_pair /taskC/output/partition_7 /taskC/output/msa_output/gpu_7

The key idea is:
  - partition_input_dir is the existing logical partition directory
  - msa_output_dir is the original output directory for that logical partition
  - the script audits msa_output_dir, reruns only missing jobs, and writes the
    new MSA outputs back into the same msa_output_dir
  - physical GPU assignment is dynamic across the provided GPU list
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionTask:
    index: int
    partition_input_dir: pathlib.Path
    msa_output_dir: pathlib.Path
    resume_dir: pathlib.Path
    label: str


@dataclass(frozen=True)
class AuditSummary:
    done: int
    todo: int
    total: int


def sanitise(name: str) -> str:
    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.")
    return "".join(ch for ch in spaceless_name if ch in allowed_chars)


def load_job_name(json_path: pathlib.Path) -> str:
    raw = json.loads(json_path.read_text())
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"{json_path} contains {len(raw)} jobs; recovery expects one job per file."
            )
        raw = raw[0]
    if "name" not in raw:
        raise ValueError(f"{json_path} does not contain a 'name' field.")
    return sanitise(raw["name"])


def audit_partition(
    expected_dir: pathlib.Path,
    output_dir: pathlib.Path,
    resume_dir: pathlib.Path,
) -> AuditSummary:
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    return AuditSummary(done=done, todo=todo, total=total)


def calc_reduced_batch_size(base_batch: int, pass_index: int, todo_count: int) -> int:
    batch = max(1, base_batch)
    for _ in range(1, pass_index):
        batch = max(1, (batch + 1) // 2)
    return max(1, min(batch, todo_count))


def format_paths(task: PartitionTask) -> str:
    return (
        f"partition_input={task.partition_input_dir} "
        f"msa_output={task.msa_output_dir}"
    )


def build_pythonpath(app_dir: pathlib.Path) -> str:
    src_dir = app_dir / "src"
    current = os.environ.get("PYTHONPATH")
    if current:
        return f"{src_dir}:{current}"
    return str(src_dir)


def run_partition_task(
    task: PartitionTask,
    gpu_id: str,
    args: argparse.Namespace,
    run_data_pipeline_path: pathlib.Path,
    mmseqs_temp_root: pathlib.Path,
    pythonpath: str,
    log_dir: pathlib.Path,
    print_lock: threading.Lock,
) -> tuple[bool, str]:
    base_batch = args.batch_size
    pass_index = 1
    max_passes = args.max_auto_reruns + 1

    while True:
        summary = audit_partition(
            expected_dir=task.partition_input_dir,
            output_dir=task.msa_output_dir,
            resume_dir=task.resume_dir,
        )
        with print_lock:
            print(
                f"[{task.label}] audit before pass {pass_index}: "
                f"done={summary.done}, missing={summary.todo}, expected={summary.total} | "
                f"{format_paths(task)}",
                flush=True,
            )

        if summary.total == 0:
            return False, f"{task.label}: no .json files found under {task.partition_input_dir}"
        if summary.todo == 0:
            return True, f"{task.label}: already complete"
        if pass_index > max_passes:
            return (
                False,
                f"{task.label}: still missing {summary.todo}/{summary.total} job(s) "
                f"after {args.max_auto_reruns} auto-reruns",
            )

        current_batch = calc_reduced_batch_size(base_batch, pass_index, summary.todo)
        log_path = log_dir / f"{task.label}_gpu{gpu_id}_pass{pass_index}.log"
        gpu_temp_dir = mmseqs_temp_root / f"gpu_{gpu_id}"
        gpu_temp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(run_data_pipeline_path),
            f"--input_dir={task.resume_dir}",
            f"--output_dir={task.msa_output_dir}",
            f"--db_dir={args.db_dir}",
            f"--mmseqs_db_dir={args.mmseqs_db_dir}",
            "--use_mmseqs_gpu",
            f"--batch_size={current_batch}",
            f"--mmseqs_n_threads={args.mmseqs_threads}",
            f"--temp_dir={gpu_temp_dir}",
        ]
        if args.template_batch_size is not None:
            cmd.append(f"--template_batch_size={args.template_batch_size}")
        if args.template_max_attempts is not None:
            cmd.append(f"--template_max_attempts={args.template_max_attempts}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONPATH"] = pythonpath

        with log_path.open("wt") as log_file:
            log_file.write(
                f"task={task.label}\n"
                f"gpu_id={gpu_id}\n"
                f"{format_paths(task)}\n"
                f"pass_index={pass_index}\n"
                f"batch_size={current_batch}\n"
                f"missing_before={summary.todo}\n"
                f"expected_total={summary.total}\n"
                f"command={' '.join(cmd)}\n\n"
            )
            status = subprocess.call(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

        post_summary = audit_partition(
            expected_dir=task.partition_input_dir,
            output_dir=task.msa_output_dir,
            resume_dir=task.resume_dir,
        )
        with print_lock:
            print(
                f"[{task.label}] GPU {gpu_id} pass {pass_index} finished "
                f"(exit={status}) -> done={post_summary.done}, missing={post_summary.todo}, "
                f"expected={post_summary.total} | log={log_path}",
                flush=True,
            )

        if post_summary.todo == 0:
            return True, f"{task.label}: recovery complete"

        pass_index += 1


def worker_loop(
    gpu_id: str,
    task_queue: queue.Queue[PartitionTask],
    args: argparse.Namespace,
    run_data_pipeline_path: pathlib.Path,
    mmseqs_temp_root: pathlib.Path,
    pythonpath: str,
    log_dir: pathlib.Path,
    results: list[tuple[PartitionTask, bool, str]],
    result_lock: threading.Lock,
    print_lock: threading.Lock,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return

        with print_lock:
            print(
                f"[{task.label}] assigned to physical GPU {gpu_id} | {format_paths(task)}",
                flush=True,
            )
        success, message = run_partition_task(
            task=task,
            gpu_id=gpu_id,
            args=args,
            run_data_pipeline_path=run_data_pipeline_path,
            mmseqs_temp_root=mmseqs_temp_root,
            pythonpath=pythonpath,
            log_dir=log_dir,
            print_lock=print_lock,
        )
        with result_lock:
            results.append((task, success, message))
        task_queue.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun failed MSA partitions while preserving existing partition outputs."
    )
    parser.add_argument(
        "--partition_pair",
        nargs=2,
        action="append",
        metavar=("PARTITION_INPUT_DIR", "MSA_OUTPUT_DIR"),
        required=True,
        help=(
            "Manual pair of existing logical partition input directory and its "
            "corresponding MSA output directory. Repeat this option for multiple "
            "failed partitions, including partitions from different tasks."
        ),
    )
    parser.add_argument("--db_dir", required=True, help="Genetic database directory.")
    parser.add_argument(
        "--mmseqs_db_dir", required=True, help="MMseqs2 database directory."
    )
    parser.add_argument(
        "--gpu_devices",
        default="0",
        help="Comma-separated physical GPU IDs to use for recovery. Default: 0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Base MSA batch size before auto-rerun reduction. Default: 256",
    )
    parser.add_argument(
        "--mmseqs_threads",
        type=int,
        default=15,
        help="MMseqs2 threads per GPU process. Default: 15",
    )
    parser.add_argument(
        "--template_batch_size",
        type=int,
        default=None,
        help="Optional template batch size override.",
    )
    parser.add_argument(
        "--template_max_attempts",
        type=int,
        default=None,
        help="Optional template retry attempts override.",
    )
    parser.add_argument(
        "--max_auto_reruns",
        type=int,
        default=3,
        help="Maximum automatic reruns per logical partition. Default: 3",
    )
    parser.add_argument(
        "--temp_dir",
        default=None,
        help=(
            "Optional base temp directory for helper files and MMseqs working files. "
            "If omitted, a temporary directory is created under the system temp area."
        ),
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help=(
            "Optional log directory. Default: ./logs/msa_recovery_<timestamp> "
            "relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--app_dir",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="Path to the alphafast1 repo. Default: scripts/..",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.mmseqs_threads < 1:
        raise SystemExit("--mmseqs_threads must be >= 1")
    if args.max_auto_reruns < 0:
        raise SystemExit("--max_auto_reruns must be >= 0")

    app_dir = pathlib.Path(args.app_dir).resolve()
    run_data_pipeline_path = app_dir / "run_data_pipeline.py"
    if not run_data_pipeline_path.is_file():
        raise SystemExit(f"run_data_pipeline.py not found under {app_dir}")

    db_dir = pathlib.Path(args.db_dir).resolve()
    mmseqs_db_dir = pathlib.Path(args.mmseqs_db_dir).resolve()
    for path in (db_dir, mmseqs_db_dir, app_dir):
        if not path.is_dir():
            raise SystemExit(f"Directory not found: {path}")
    args.db_dir = str(db_dir)
    args.mmseqs_db_dir = str(mmseqs_db_dir)

    gpu_devices = [gpu.strip() for gpu in args.gpu_devices.split(",") if gpu.strip()]
    if not gpu_devices:
        raise SystemExit("--gpu_devices must contain at least one GPU ID")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.log_dir is None:
        log_dir = pathlib.Path.cwd() / "logs" / f"msa_recovery_{timestamp}"
    else:
        log_dir = pathlib.Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.temp_dir is None:
        temp_root = pathlib.Path(
            tempfile.mkdtemp(prefix="alphafast_msa_recovery_")
        ).resolve()
    else:
        temp_root = pathlib.Path(args.temp_dir).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
    resume_root = temp_root / "resume"
    mmseqs_temp_root = temp_root / "mmseqs"
    resume_root.mkdir(parents=True, exist_ok=True)
    mmseqs_temp_root.mkdir(parents=True, exist_ok=True)

    seen_output_dirs: set[pathlib.Path] = set()
    tasks: list[PartitionTask] = []
    for index, (partition_input_str, msa_output_str) in enumerate(args.partition_pair):
        partition_input_dir = pathlib.Path(partition_input_str).resolve()
        msa_output_dir = pathlib.Path(msa_output_str).resolve()

        if not partition_input_dir.is_dir():
            raise SystemExit(f"Partition input directory not found: {partition_input_dir}")
        if msa_output_dir in seen_output_dirs:
            raise SystemExit(
                f"Duplicate MSA output directory is not allowed: {msa_output_dir}"
            )
        seen_output_dirs.add(msa_output_dir)

        label = f"task{index:02d}"
        tasks.append(
            PartitionTask(
                index=index,
                partition_input_dir=partition_input_dir,
                msa_output_dir=msa_output_dir,
                resume_dir=resume_root / label,
                label=label,
            )
        )

    pythonpath = build_pythonpath(app_dir)

    print("==========================================")
    print("AlphaFast MSA Partition Recovery")
    print("==========================================")
    print(f"App dir:           {app_dir}")
    print(f"DB dir:            {db_dir}")
    print(f"MMseqs dir:        {mmseqs_db_dir}")
    print(f"GPU devices:       {','.join(gpu_devices)}")
    print(f"Base batch size:   {args.batch_size}")
    print(f"MMseqs threads:    {args.mmseqs_threads}")
    print(f"Max auto reruns:   {args.max_auto_reruns}")
    print(f"Temp root:         {temp_root}")
    print(f"Log dir:           {log_dir}")
    print(f"Partition pairs:   {len(tasks)}")
    print("==========================================")
    for task in tasks:
        print(f"{task.label}: {format_paths(task)}")
    print("")

    initial_audits: list[tuple[PartitionTask, AuditSummary]] = []
    for task in tasks:
        summary = audit_partition(
            expected_dir=task.partition_input_dir,
            output_dir=task.msa_output_dir,
            resume_dir=task.resume_dir,
        )
        initial_audits.append((task, summary))
        print(
            f"{task.label}: initial audit done={summary.done}, "
            f"missing={summary.todo}, expected={summary.total}",
            flush=True,
        )

    pending_tasks = [
        task for task, summary in sorted(
            initial_audits, key=lambda item: item[1].todo, reverse=True
        )
        if summary.todo > 0
    ]
    if not pending_tasks:
        print("")
        print("All provided partitions already have complete MSA outputs. Nothing to do.")
        return 0

    task_queue: queue.Queue[PartitionTask] = queue.Queue()
    for task in pending_tasks:
        task_queue.put(task)

    results: list[tuple[PartitionTask, bool, str]] = []
    result_lock = threading.Lock()
    print_lock = threading.Lock()
    threads: list[threading.Thread] = []

    for gpu_id in gpu_devices:
        thread = threading.Thread(
            target=worker_loop,
            args=(
                gpu_id,
                task_queue,
                args,
                run_data_pipeline_path,
                mmseqs_temp_root,
                pythonpath,
                log_dir,
                results,
                result_lock,
                print_lock,
            ),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results_by_index = {task.index: (success, message) for task, success, message in results}
    failures = 0
    print("")
    print("==========================================")
    print("Recovery Summary")
    print("==========================================")
    for task in tasks:
        if task.index in results_by_index:
            success, message = results_by_index[task.index]
        else:
            success = True
            message = f"{task.label}: already complete"
        status = "OK" if success else "FAILED"
        print(f"{status}: {message}")
        if not success:
            failures += 1

    print("")
    print(f"Logs written under: {log_dir}")
    print(f"Temp root retained at: {temp_root}")
    print("==========================================")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
