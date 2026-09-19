#!/usr/bin/env python3
"""
run_benchmark.py — Chạy toàn bộ benchmark và xuất kết quả ra CSV
=================================================================

Script này chạy các Variant 1, 2 (incremental), 2 (non-incremental) và 3
trên benchmark_v1, thu thập kết quả và ghi tự động vào CSV trong results/.

Cách dùng
---------
    # Chạy tất cả variant trên toàn bộ benchmark:
    python3 experiments/run_benchmark.py

    # Chỉ chạy một số variant:
    python3 experiments/run_benchmark.py --variants 1 2

    # Giới hạn thư mục instance:
    python3 experiments/run_benchmark.py --input-dir data/benchmark_v1/easy

    # Đổi timeout (mặc định 600s):
    python3 experiments/run_benchmark.py --timeout 300

    # Đổi thư mục output:
    python3 experiments/run_benchmark.py --output-dir results/run_001

Các file CSV được ghi:
    results/feasibility.csv        (Variant 1)
    results/optimum_binary.csv     (Variant 2 incremental — binary search)
    results/optimum_linear.csv     (Variant 2 non-incremental — binary search)
    results/maxmin.csv             (Variant 3 — CEGAR max-min)
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# Cho phép import từ thư mục models/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from sat_variant1 import (
    AlienTilesSAT,
    load_instances_from_dir,
)
from sat_variant2 import AlienTilesMinSAT
from sat_variant2_incremental import AlienTilesMinIncrSAT
from sat_variant3 import AlienTilesMaxMinSAT


# =====================================================================
#  Timeout helper (Unix — dùng SIGALRM)
# =====================================================================

class _TimeoutError(Exception):
    pass


class timeout_ctx:
    """Context manager: raise _TimeoutError after `seconds` seconds."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise _TimeoutError()

    def __enter__(self):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return False


# =====================================================================
#  CSV helpers
# =====================================================================

SOLVER_COLUMNS = [
    "instance", "N", "c",
    "variables", "clauses",
    "runtime_s", "sat_calls",
    "total_clicks", "optimal_clicks",
    "status",
]

MAXMIN_COLUMNS = [
    "config", "N", "c",
    "maxmin_clicks",
    "targets_checked", "cegar_iterations", "variant2_calls",
    "runtime_s",
    "status",
    "hardest_target",
]


def _open_csv(path: Path, columns: list) -> tuple:
    """Mở (hoặc nối vào) file CSV; ghi header nếu file mới."""
    is_new = not path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    # Gắn stream để có thể flush sau mỗi dòng
    writer.stream = fh
    return writer, fh


def _write_row(writer, row: dict):
    """Ghi một dòng kết quả, flush ngay để không mất dữ liệu khi crash."""
    writer.writerow(row)
    writer.stream.flush()


# =====================================================================
#  Per-variant runners
# =====================================================================

def run_feasibility(inst: dict, timeout_s: int) -> dict:
    """Variant 1: feasibility check."""
    N, c, target = inst["N"], inst["c"], inst["target"]
    row = dict(instance=inst["name"], N=N, c=c,
               variables=None, clauses=None,
               runtime_s=None, sat_calls=1,
               total_clicks=None, optimal_clicks=None,
               status="ERROR")
    t0 = time.perf_counter()
    try:
        with timeout_ctx(timeout_s):
            sat = AlienTilesSAT(N, c, target)
            solution = sat.solve()
            row["runtime_s"] = round(time.perf_counter() - t0, 4)
            row["variables"] = sat.stats["vars"]
            row["clauses"] = sat.stats["clauses"]
            if solution is None:
                row["status"] = "UNSAT"
            else:
                row["total_clicks"] = sum(sum(r) for r in solution)
                row["status"] = "OK"
    except _TimeoutError:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "TIMEOUT"
    except Exception:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "ERROR"
        traceback.print_exc()
    return row


def run_optimum_binary(inst: dict, timeout_s: int) -> dict:
    """Variant 2 incremental (binary search): minimisation."""
    N, c, target = inst["N"], inst["c"], inst["target"]
    row = dict(instance=inst["name"], N=N, c=c,
               variables=None, clauses=None,
               runtime_s=None, sat_calls=None,
               total_clicks=None, optimal_clicks=None,
               status="ERROR")
    t0 = time.perf_counter()
    try:
        with timeout_ctx(timeout_s):
            solver = AlienTilesMinIncrSAT(N, c, target)
            solution, min_total = solver.solve()
            row["runtime_s"] = round(time.perf_counter() - t0, 4)
            row["variables"] = solver.stats["vars"]
            row["clauses"] = solver.stats["clauses"]
            row["sat_calls"] = solver.stats["sat_calls"]
            if solution is None:
                row["status"] = "UNSAT"
            else:
                row["total_clicks"] = min_total
                row["optimal_clicks"] = min_total
                row["status"] = "OK"
    except _TimeoutError:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "TIMEOUT"
    except Exception:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "ERROR"
        traceback.print_exc()
    return row


def run_optimum_linear(inst: dict, timeout_s: int) -> dict:
    """Variant 2 non-incremental (binary search): minimisation."""
    N, c, target = inst["N"], inst["c"], inst["target"]
    row = dict(instance=inst["name"], N=N, c=c,
               variables=None, clauses=None,
               runtime_s=None, sat_calls=None,
               total_clicks=None, optimal_clicks=None,
               status="ERROR")
    t0 = time.perf_counter()
    try:
        with timeout_ctx(timeout_s):
            solver = AlienTilesMinSAT(N, c, target)
            solution, min_total = solver.solve()
            row["runtime_s"] = round(time.perf_counter() - t0, 4)
            row["variables"] = solver.stats["vars"]
            row["clauses"] = solver.stats["clauses"]
            row["sat_calls"] = solver.stats["sat_calls"]
            if solution is None:
                row["status"] = "UNSAT"
            else:
                row["total_clicks"] = min_total
                row["optimal_clicks"] = min_total
                row["status"] = "OK"
    except _TimeoutError:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "TIMEOUT"
    except Exception:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "ERROR"
        traceback.print_exc()
    return row


def run_maxmin(N: int, c: int, timeout_s: int) -> dict:
    """Variant 3: max-min (CEGAR)."""
    config = f"{N}x{N}_c{c}"
    row = dict(config=config, N=N, c=c,
               maxmin_clicks=None,
               targets_checked=None, cegar_iterations=None, variant2_calls=None,
               runtime_s=None,
               status="ERROR",
               hardest_target=None)
    t0 = time.perf_counter()
    try:
        with timeout_ctx(timeout_s):
            solver = AlienTilesMaxMinSAT(N, c)
            target, solution, max_min = solver.solve()
            row["runtime_s"] = round(time.perf_counter() - t0, 4)
            row["targets_checked"] = solver.stats["targets_checked"]
            row["cegar_iterations"] = solver.stats["iterations"]
            row["variant2_calls"] = solver.stats["var2_calls"]
            if target is None:
                row["status"] = "NO_SOLUTION"
            else:
                row["maxmin_clicks"] = max_min
                row["hardest_target"] = str(target)
                row["status"] = "OK"
    except _TimeoutError:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "TIMEOUT"
    except Exception:
        row["runtime_s"] = round(time.perf_counter() - t0, 4)
        row["status"] = "ERROR"
        traceback.print_exc()
    return row


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chạy benchmark Alien Tiles và xuất kết quả ra CSV")
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "data" / "benchmark_v1"),
        help="Thư mục chứa instance JSON (mặc định: data/benchmark_v1)")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results"),
        help="Thư mục ghi CSV (mặc định: results/)")
    parser.add_argument(
        "--variants", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Variant cần chạy: 1=feasibility, 2=optimum(binary+linear), "
             "3=maxmin  (mặc định: 1 2 3)")
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout mỗi lần giải (giây, mặc định: 600)")
    parser.add_argument(
        "--maxmin-sizes", nargs="+", default=None,
        metavar="N:c",
        help="Cấu hình cho variant 3, dạng N:c (vd: 4:2 4:3 5:2). "
             "Mặc định: tất cả (N,c) duy nhất trong benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nạp instances
    print(f"Đang nạp instances từ: {args.input_dir}")
    instances = load_instances_from_dir(args.input_dir)
    print(f"Tìm thấy {len(instances)} instances.\n")

    # ------------------------------------------------------------------
    # Variant 1: Feasibility
    # ------------------------------------------------------------------
    if 1 in args.variants:
        csv_path = out_dir / "feasibility.csv"
        writer, fh = _open_csv(csv_path, SOLVER_COLUMNS)
        print(f"{'='*60}")
        print(f"[Variant 1] Feasibility  →  {csv_path}")
        print(f"{'='*60}")
        for inst in instances:
            print(f"  {inst['name']:50s}", end="", flush=True)
            row = run_feasibility(inst, args.timeout)
            _write_row(writer, row)
            print(f"{row['status']:8s}  {row['runtime_s']}s"
                  f"  vars={row['variables']}  cls={row['clauses']}")
        fh.close()
        print()

    # ------------------------------------------------------------------
    # Variant 2: Optimum — incremental binary + non-incremental binary
    # ------------------------------------------------------------------
    if 2 in args.variants:
        # 2a. Incremental (khuyến nghị)
        csv_path = out_dir / "optimum_binary.csv"
        writer, fh = _open_csv(csv_path, SOLVER_COLUMNS)
        print(f"{'='*60}")
        print(f"[Variant 2-incr] Optimum binary search  →  {csv_path}")
        print(f"{'='*60}")
        for inst in instances:
            print(f"  {inst['name']:50s}", end="", flush=True)
            row = run_optimum_binary(inst, args.timeout)
            _write_row(writer, row)
            suffix = f"  opt={row['optimal_clicks']}" if row["optimal_clicks"] is not None else ""
            print(f"{row['status']:8s}  {row['runtime_s']}s"
                  f"  calls={row['sat_calls']}{suffix}")
        fh.close()
        print()

        # 2b. Non-incremental (để so sánh)
        csv_path = out_dir / "optimum_linear.csv"
        writer, fh = _open_csv(csv_path, SOLVER_COLUMNS)
        print(f"{'='*60}")
        print(f"[Variant 2-linear] Optimum binary search (non-incr)  →  {csv_path}")
        print(f"{'='*60}")
        for inst in instances:
            print(f"  {inst['name']:50s}", end="", flush=True)
            row = run_optimum_linear(inst, args.timeout)
            _write_row(writer, row)
            suffix = f"  opt={row['optimal_clicks']}" if row["optimal_clicks"] is not None else ""
            print(f"{row['status']:8s}  {row['runtime_s']}s"
                  f"  calls={row['sat_calls']}{suffix}")
        fh.close()
        print()

    # ------------------------------------------------------------------
    # Variant 3: Max-Min (CEGAR)
    # ------------------------------------------------------------------
    if 3 in args.variants:
        if args.maxmin_sizes:
            configs = []
            for token in args.maxmin_sizes:
                n_str, c_str = token.split(":")
                configs.append((int(n_str), int(c_str)))
        else:
            seen: set = set()
            configs = []
            for inst in instances:
                key = (inst["N"], inst["c"])
                if key not in seen:
                    seen.add(key)
                    configs.append(key)
            configs.sort()

        csv_path = out_dir / "maxmin.csv"
        writer, fh = _open_csv(csv_path, MAXMIN_COLUMNS)
        print(f"{'='*60}")
        print(f"[Variant 3] Max-Min CEGAR  →  {csv_path}")
        print(f"{'='*60}")
        for N, c in configs:
            config = f"{N}x{N}_c{c}"
            print(f"  {config:20s}", end="", flush=True)
            row = run_maxmin(N, c, args.timeout)
            _write_row(writer, row)
            print(f"{row['status']:8s}  {row['runtime_s']}s"
                  f"  maxmin={row['maxmin_clicks']}")
        fh.close()
        print()

    print(f"Hoàn tất. Kết quả CSV đã ghi vào: {args.output_dir}")


if __name__ == "__main__":
    main()
