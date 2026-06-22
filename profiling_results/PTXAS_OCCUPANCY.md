# gsplat Rasterize PTXAS / Occupancy

- PTXAS log: `/data2/cse12311966/projects/gsplat-lib/build_ptxas_verbose_sm70.log`
- extracted cubin dir: `/tmp/gsplat_occ`
- audit artifacts in this repo: [`script/occupancy_gsplat_raster.py`](../script/occupancy_gsplat_raster.py), [`profiling_results/ptxas_occupancy_rows.csv`](ptxas_occupancy_rows.csv), [`profiling_results/ptxas_rasterize_all_cdims.csv`](ptxas_rasterize_all_cdims.csv)
- target CDIMs: `[3, 16, 19, 32]`
- block size: `16 x 16 = 256` threads
- register counts are parsed from PTXAS and cross-checked against `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)` for the same mangled cubin symbol.
- the build log also contains many larger CDIM instantiations with `255` registers and spill traffic; they are outside this table's target scope and are intentionally excluded.
- occupancy is computed with `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the real cubin function.
- a crude `grep -B8` window is ambiguous because the build log also contains 2DGS kernels; the audit script instead matches exact 3DGS mangled symbols and verifies them against the cubin and driver attributes.

| CDIM | Mode | Regs/thread | Stack B | Spill store B | Spill load B | Static smem B | Dynamic smem B | Blocks/SM | Occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | fwd | 40 | 0 | 0 | 0 | 0 | 7168 | 6 | 75.0% |
| 16 | fwd | 56 | 0 | 0 | 0 | 0 | 7168 | 4 | 50.0% |
| 19 | fwd | 72 | 0 | 0 | 0 | 0 | 7168 | 3 | 37.5% |
| 32 | fwd | 64 | 0 | 0 | 0 | 0 | 7168 | 4 | 50.0% |
| 3 | bwd | 74 | 0 | 0 | 0 | 0 | 10240 | 3 | 37.5% |
| 16 | bwd | 112 | 0 | 0 | 0 | 0 | 23552 | 2 | 25.0% |
| 19 | bwd | 122 | 0 | 0 | 0 | 0 | 26624 | 2 | 25.0% |
| 32 | bwd | 162 | 0 | 0 | 0 | 0 | 39936 | 1 | 12.5% |

## Spill Findings

No target CDIM row reported stack/spill traffic.

## Non-Target Spill Rows

The verbose build log does contain high-register spill-heavy rasterize instantiations, but they are not CDIM 3/16/19/32. They come from larger template instantiations compiled by gsplat:

- CDIM 128 bwd: 255 regs/thread, stack 760 B, spill stores 3664 B, spill loads 4632 B (PTXAS lines 492-495).
- CDIM 129 bwd: 255 regs/thread, stack 784 B, spill stores 3796 B, spill loads 4768 B (PTXAS lines 488-491).
- CDIM 256 bwd: 255 regs/thread, stack 2416 B, spill stores 3836 B, spill loads 3156 B (PTXAS lines 484-487).
- CDIM 257 bwd: 255 regs/thread, stack 2448 B, spill stores 5020 B, spill loads 4344 B (PTXAS lines 480-483).
- CDIM 512 bwd: 255 regs/thread, stack 6704 B, spill stores 13800 B, spill loads 24728 B (PTXAS lines 476-479).
- CDIM 513 bwd: 255 regs/thread, stack 6736 B, spill stores 27452 B, spill loads 27940 B (PTXAS lines 472-475).
- CDIM 129 fwd: 255 regs/thread, stack 872 B, spill stores 1136 B, spill loads 1728 B (PTXAS lines 406-409).
- CDIM 256 fwd: 255 regs/thread, stack 120 B, spill stores 196 B, spill loads 276 B (PTXAS lines 402-405).
- CDIM 257 fwd: 255 regs/thread, stack 3104 B, spill stores 5076 B, spill loads 8248 B (PTXAS lines 398-401).
- CDIM 512 fwd: 255 regs/thread, stack 1184 B, spill stores 2248 B, spill loads 3352 B (PTXAS lines 394-397).
- CDIM 513 fwd: 255 regs/thread, stack 6728 B, spill stores 12356 B, spill loads 20644 B (PTXAS lines 390-393).

## Audit Trail

Each target row records the PTXAS entry/usage line numbers and the cubin that supplied the driver API function. See `ptxas_occupancy_rows.csv` for exact mangled symbols. See `ptxas_rasterize_all_cdims.csv` for the full ordinary rasterize CDIM mapping parsed from the same build log.
