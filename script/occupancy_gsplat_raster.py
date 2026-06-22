#!/usr/bin/env python3
"""Audit gsplat 3DGS rasterize PTXAS resources and occupancy.

This script intentionally keeps the two data sources explicit:

1. PTXAS resource usage is parsed from a verbose nvcc build log.
2. Occupancy is computed from the real cubin symbols with the CUDA Driver API.

The rows are joined by the exact mangled kernel symbol. If PTXAS and the cubin
do not agree on registers/thread for a target kernel, the script fails.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import re
import subprocess
from ctypes import byref, c_char_p, c_int, c_size_t, c_uint, c_void_p
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TARGET_CDIMS = (3, 16, 19, 32)
TILE_SIZE = 16
BLOCK_THREADS = TILE_SIZE * TILE_SIZE
MAX_THREADS_PER_SM_SM70 = 2048

ENTRY_RE = re.compile(r"ptxas info\s+: Compiling entry function '([^']+)'")
STACK_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
USED_RE = re.compile(r"Used (\d+) registers(?:, (.*))?$")
SMEM_RE = re.compile(r"(\d+) bytes smem")
CMEM_RE = re.compile(r"(\d+) bytes cmem\[(\d+)\]")
FWD_RE = re.compile(
    r"^_ZN6gsplat35rasterize_to_pixels_3dgs_fwd_kernelILj(\d+)EfEEv.*$"
)
BWD_RE = re.compile(
    r"^_ZN6gsplat35rasterize_to_pixels_3dgs_bwd_kernelILj(\d+)EfEEv.*$"
)

# CUDA function attributes.
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4


def classify_symbol(symbol: str) -> Tuple[Optional[str], Optional[int]]:
    match = FWD_RE.match(symbol)
    if match:
        return "fwd", int(match.group(1))
    match = BWD_RE.match(symbol)
    if match:
        return "bwd", int(match.group(1))
    return None, None


def parse_ptxas_log_all(log_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    current: Optional[Dict] = None

    for lineno, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
        entry = ENTRY_RE.search(line)
        if entry:
            symbol = entry.group(1)
            kind, cdim = classify_symbol(symbol)
            current = {
                "symbol": symbol,
                "kind": kind,
                "cdim": cdim,
                "entry_line": lineno,
                "stack_frame_bytes": None,
                "spill_stores_bytes": None,
                "spill_loads_bytes": None,
            }
            continue

        if current is None:
            continue

        stack = STACK_RE.search(line)
        if stack:
            (
                current["stack_frame_bytes"],
                current["spill_stores_bytes"],
                current["spill_loads_bytes"],
            ) = [int(value) for value in stack.groups()]
            continue

        used = USED_RE.search(line)
        if used:
            current["ptxas_regs_per_thread"] = int(used.group(1))
            rest = used.group(2) or ""
            smem = SMEM_RE.search(rest)
            current["ptxas_static_smem_bytes"] = int(smem.group(1)) if smem else 0
            current["ptxas_cmem_bytes"] = sum(
                int(size) for size, _index in CMEM_RE.findall(rest)
            )
            current["used_line"] = lineno

            if current["kind"] is not None:
                rows.append(current)
            current = None

    return rows


def filter_target_ptxas_rows(all_rows: Iterable[Dict]) -> Dict[Tuple[str, int], Dict]:
    rows: Dict[Tuple[str, int], Dict] = {}
    for row in all_rows:
        if row["cdim"] not in TARGET_CDIMS:
            continue
        key = (row["kind"], row["cdim"])
        if key in rows:
            raise RuntimeError(f"Duplicate PTXAS row for {key}")
        rows[key] = row

    missing = [
        (kind, cdim)
        for kind in ("fwd", "bwd")
        for cdim in TARGET_CDIMS
        if (kind, cdim) not in rows
    ]
    if missing:
        raise RuntimeError(f"Missing target kernels in PTXAS log: {missing}")
    return rows


def parse_ptxas_log(log_path: Path) -> Dict[Tuple[str, int], Dict]:
    return filter_target_ptxas_rows(parse_ptxas_log_all(log_path))


def dynamic_smem_bytes(kind: str, cdim: int) -> int:
    """Match gsplat RasterizeToPixels3DGS{Fwd,Bwd}.cu launch shmem_size."""
    base = BLOCK_THREADS * (4 + 12 + 12)
    if kind == "fwd":
        return base
    return base + BLOCK_THREADS * 4 * cdim


def check(code: int, label: str) -> None:
    if code != 0:
        raise RuntimeError(f"{label} failed with CUDA error {code}")


def find_cubin_symbols(cubin_dir: Path) -> Dict[Tuple[str, int], Dict]:
    symbols: Dict[Tuple[str, int], Dict] = {}
    for cubin in sorted(cubin_dir.glob("*.cubin")):
        proc = subprocess.run(
            ["strings", str(cubin)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        for line in proc.stdout.splitlines():
            kind, cdim = classify_symbol(line)
            if kind is None or cdim not in TARGET_CDIMS:
                continue
            key = (kind, cdim)
            if key not in symbols:
                symbols[key] = {"cubin": cubin, "symbol": line}

    missing = [
        (kind, cdim)
        for kind in ("fwd", "bwd")
        for cdim in TARGET_CDIMS
        if (kind, cdim) not in symbols
    ]
    if missing:
        raise RuntimeError(f"Missing target kernels in extracted cubins: {missing}")
    return symbols


class CudaDriver:
    def __init__(self) -> None:
        self.libcuda = ctypes.CDLL("libcuda.so.1")

        self.cu_init = self.libcuda.cuInit
        self.cu_init.argtypes = [c_uint]
        self.cu_device_get = self.libcuda.cuDeviceGet
        self.cu_device_get.argtypes = [ctypes.POINTER(c_int), c_int]
        self.cu_ctx_create = self.libcuda.cuCtxCreate_v2
        self.cu_ctx_create.argtypes = [ctypes.POINTER(c_void_p), c_uint, c_int]
        self.cu_module_load = self.libcuda.cuModuleLoad
        self.cu_module_load.argtypes = [ctypes.POINTER(c_void_p), c_char_p]
        self.cu_module_get_function = self.libcuda.cuModuleGetFunction
        self.cu_module_get_function.argtypes = [
            ctypes.POINTER(c_void_p),
            c_void_p,
            c_char_p,
        ]
        self.cu_func_get_attribute = self.libcuda.cuFuncGetAttribute
        self.cu_func_get_attribute.argtypes = [
            ctypes.POINTER(c_int),
            c_int,
            c_void_p,
        ]
        self.cu_occupancy = self.libcuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
        self.cu_occupancy.argtypes = [
            ctypes.POINTER(c_int),
            c_void_p,
            c_int,
            c_size_t,
        ]

        check(self.cu_init(0), "cuInit")
        dev = c_int()
        check(self.cu_device_get(byref(dev), 0), "cuDeviceGet")
        ctx = c_void_p()
        check(self.cu_ctx_create(byref(ctx), 0, dev), "cuCtxCreate")
        self.modules: Dict[Path, c_void_p] = {}

    def load_function(self, cubin: Path, symbol: str) -> c_void_p:
        if cubin not in self.modules:
            module = c_void_p()
            check(self.cu_module_load(byref(module), str(cubin).encode()), f"cuModuleLoad {cubin.name}")
            self.modules[cubin] = module

        func = c_void_p()
        check(
            self.cu_module_get_function(
                byref(func), self.modules[cubin], symbol.encode()
            ),
            f"cuModuleGetFunction {symbol}",
        )
        return func

    def function_attribute(self, func: c_void_p, attr: int) -> int:
        value = c_int()
        check(self.cu_func_get_attribute(byref(value), attr, func), f"cuFuncGetAttribute {attr}")
        return int(value.value)

    def occupancy_blocks(self, func: c_void_p, block_threads: int, dynamic_smem: int) -> int:
        active_blocks = c_int()
        check(
            self.cu_occupancy(byref(active_blocks), func, block_threads, dynamic_smem),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        return int(active_blocks.value)


def build_rows(ptxas_rows: Dict[Tuple[str, int], Dict], cubin_rows: Dict[Tuple[str, int], Dict]) -> List[Dict]:
    driver = CudaDriver()
    rows: List[Dict] = []

    for kind in ("fwd", "bwd"):
        for cdim in TARGET_CDIMS:
            key = (kind, cdim)
            ptxas = ptxas_rows[key]
            cubin = cubin_rows[key]

            if ptxas["symbol"] != cubin["symbol"]:
                raise RuntimeError(
                    f"Symbol mismatch for {key}: PTXAS={ptxas['symbol']} cubin={cubin['symbol']}"
                )

            func = driver.load_function(cubin["cubin"], cubin["symbol"])
            driver_regs = driver.function_attribute(func, CU_FUNC_ATTRIBUTE_NUM_REGS)
            driver_static_smem = driver.function_attribute(
                func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
            )
            driver_const_bytes = driver.function_attribute(
                func, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
            )
            driver_local_bytes = driver.function_attribute(
                func, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            )
            driver_max_threads = driver.function_attribute(
                func, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            )

            if driver_regs != ptxas["ptxas_regs_per_thread"]:
                raise RuntimeError(
                    f"Register mismatch for {key}: PTXAS={ptxas['ptxas_regs_per_thread']} "
                    f"driver={driver_regs}"
                )
            if driver_static_smem != ptxas["ptxas_static_smem_bytes"]:
                raise RuntimeError(
                    f"Static smem mismatch for {key}: PTXAS={ptxas['ptxas_static_smem_bytes']} "
                    f"driver={driver_static_smem}"
                )

            dynamic_smem = dynamic_smem_bytes(kind, cdim)
            active_blocks = driver.occupancy_blocks(func, BLOCK_THREADS, dynamic_smem)
            occupancy = active_blocks * BLOCK_THREADS / MAX_THREADS_PER_SM_SM70

            rows.append(
                {
                    "kind": kind,
                    "cdim": cdim,
                    "ptxas_regs_per_thread": ptxas["ptxas_regs_per_thread"],
                    "driver_regs_per_thread": driver_regs,
                    "stack_frame_bytes": ptxas["stack_frame_bytes"],
                    "spill_stores_bytes": ptxas["spill_stores_bytes"],
                    "spill_loads_bytes": ptxas["spill_loads_bytes"],
                    "ptxas_static_smem_bytes": ptxas["ptxas_static_smem_bytes"],
                    "dynamic_smem_bytes": dynamic_smem,
                    "driver_static_smem_bytes": driver_static_smem,
                    "driver_const_bytes": driver_const_bytes,
                    "driver_local_bytes": driver_local_bytes,
                    "driver_max_threads_per_block": driver_max_threads,
                    "active_blocks_per_sm": active_blocks,
                    "occupancy": occupancy,
                    "cubin": cubin["cubin"].name,
                    "ptxas_entry_line": ptxas["entry_line"],
                    "ptxas_used_line": ptxas["used_line"],
                    "symbol": ptxas["symbol"],
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_all_ptxas_csv(path: Path, rows: Iterable[Dict]) -> None:
    out_rows = []
    for row in sorted(rows, key=lambda item: (item["kind"], item["cdim"])):
        out_rows.append(
            {
                "kind": row["kind"],
                "cdim": row["cdim"],
                "ptxas_regs_per_thread": row["ptxas_regs_per_thread"],
                "stack_frame_bytes": row["stack_frame_bytes"],
                "spill_stores_bytes": row["spill_stores_bytes"],
                "spill_loads_bytes": row["spill_loads_bytes"],
                "ptxas_static_smem_bytes": row["ptxas_static_smem_bytes"],
                "ptxas_cmem_bytes": row["ptxas_cmem_bytes"],
                "ptxas_entry_line": row["entry_line"],
                "ptxas_used_line": row["used_line"],
                "symbol": row["symbol"],
            }
        )
    write_csv(path, out_rows)


def markdown_table(rows: List[Dict]) -> str:
    headers = [
        "CDIM",
        "Mode",
        "Regs/thread",
        "Stack B",
        "Spill store B",
        "Spill load B",
        "Static smem B",
        "Dynamic smem B",
        "Blocks/SM",
        "Occupancy",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cdim"]),
                    row["kind"],
                    str(row["ptxas_regs_per_thread"]),
                    str(row["stack_frame_bytes"]),
                    str(row["spill_stores_bytes"]),
                    str(row["spill_loads_bytes"]),
                    str(row["ptxas_static_smem_bytes"]),
                    str(row["dynamic_smem_bytes"]),
                    str(row["active_blocks_per_sm"]),
                    f"{row['occupancy'] * 100.0:.1f}%",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(
    path: Path,
    rows: List[Dict],
    all_ptxas_rows: List[Dict],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spill_rows = [
        row
        for row in rows
        if row["spill_stores_bytes"] or row["spill_loads_bytes"] or row["stack_frame_bytes"]
    ]
    lines = [
        "# gsplat Rasterize PTXAS / Occupancy",
        "",
        f"- PTXAS log: `{args.ptxas_log}`",
        f"- extracted cubin dir: `{args.cubin_dir}`",
        f"- target CDIMs: `{list(TARGET_CDIMS)}`",
        f"- block size: `{TILE_SIZE} x {TILE_SIZE} = {BLOCK_THREADS}` threads",
        "- register counts are parsed from PTXAS and cross-checked against `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)` for the same mangled cubin symbol.",
        "- the build log also contains many larger CDIM instantiations with `255` registers and spill traffic; they are outside this table's target scope and are intentionally excluded.",
        "- occupancy is computed with `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the real cubin function.",
        "",
        markdown_table(rows),
        "",
    ]
    if spill_rows:
        lines.extend(
            [
                "## Spill Findings",
                "",
                "The target CDIM rows above include nonzero stack/spill values. These are local-memory spills reported by PTXAS and should be treated as a first-order performance signal.",
                "",
            ]
        )
        for row in spill_rows:
            lines.append(
                f"- CDIM {row['cdim']} {row['kind']}: "
                f"{row['ptxas_regs_per_thread']} regs/thread, "
                f"stack {row['stack_frame_bytes']} B, "
                f"spill stores {row['spill_stores_bytes']} B, "
                f"spill loads {row['spill_loads_bytes']} B."
            )
        lines.append("")
    else:
        lines.extend(["## Spill Findings", "", "No target CDIM row reported stack/spill traffic.", ""])

    large_spill_rows = [
        row
        for row in sorted(all_ptxas_rows, key=lambda item: (item["kind"], item["cdim"]))
        if row["cdim"] not in TARGET_CDIMS
        and (row["spill_stores_bytes"] or row["spill_loads_bytes"] or row["stack_frame_bytes"])
    ]
    if large_spill_rows:
        lines.extend(
            [
                "## Non-Target Spill Rows",
                "",
                "The verbose build log does contain high-register spill-heavy rasterize instantiations, but they are not CDIM 3/16/19/32. They come from larger template instantiations compiled by gsplat:",
                "",
            ]
        )
        for row in large_spill_rows:
            lines.append(
                f"- CDIM {row['cdim']} {row['kind']}: "
                f"{row['ptxas_regs_per_thread']} regs/thread, "
                f"stack {row['stack_frame_bytes']} B, "
                f"spill stores {row['spill_stores_bytes']} B, "
                f"spill loads {row['spill_loads_bytes']} B "
                f"(PTXAS lines {row['entry_line']}-{row['used_line']})."
            )
        lines.append("")

    lines.extend(
        [
            "## Audit Trail",
            "",
            "Each target row records the PTXAS entry/usage line numbers and the cubin that supplied the driver API function. See `ptxas_occupancy_rows.csv` for exact mangled symbols. See `ptxas_rasterize_all_cdims.csv` for the full ordinary rasterize CDIM mapping parsed from the same build log.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ptxas-log",
        required=True,
        help="Verbose nvcc build log captured with -Xptxas -v.",
    )
    parser.add_argument(
        "--cubin-dir",
        required=True,
        help="Directory containing extracted sm_70 cubins from gsplat/csrc.so.",
    )
    parser.add_argument(
        "--csv-out",
        default="profiling_results/ptxas_occupancy_rows.csv",
    )
    parser.add_argument(
        "--all-ptxas-csv-out",
        default="profiling_results/ptxas_rasterize_all_cdims.csv",
    )
    parser.add_argument(
        "--md-out",
        default="profiling_results/PTXAS_OCCUPANCY.md",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    all_ptxas_rows = parse_ptxas_log_all(Path(args.ptxas_log))
    ptxas_rows = filter_target_ptxas_rows(all_ptxas_rows)
    cubin_rows = find_cubin_symbols(Path(args.cubin_dir))
    rows = build_rows(ptxas_rows, cubin_rows)

    write_csv(Path(args.csv_out), rows)
    write_all_ptxas_csv(Path(args.all_ptxas_csv_out), all_ptxas_rows)
    write_markdown(Path(args.md_out), rows, all_ptxas_rows, args)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))

    print(markdown_table(rows))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.all_ptxas_csv_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
