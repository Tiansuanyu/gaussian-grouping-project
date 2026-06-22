# Task B Profiling Summary

## Current Status

Clean timing runs have now been collected on an idle GPU with the fixed protocol. The earlier contaminated timing data is kept as historical diagnostics only; final backend ranking below uses the clean runs.

The clean result is:

- Native `<19u>` gsplat kernels compile, are detected by the profiling script, and are functionally equivalent to the shared split-raster output.
- On the r2 workload, native 19-channel single-pass is clearly fastest.
- On the r1 workload, native 19-channel single-pass is also fastest overall, but its margin over shared-geometry split-raster is small.
- The original padded single-pass negative result is therefore closed: the `19 -> 32` padding was the main cause of the slowdown in this setup.

## What Was Fixed

`script/profile_render_backends.py` now separates timing from kernel decomposition:

- Timing uses only CUDA events.
- `torch.profiler` is disabled by default and only runs with `--profile`.
- Backends/modes are measured in an interleaved order instead of one backend block at a time.
- Multiple views are supported via `--num_views` and `--view_stride`.
- Defaults are now `50` warmup rounds and `200` measured samples per backend/mode/view. The clean-run commands below use `10` views x `20` samples to keep the same total sample count while covering view variation.
- Reports include median, mean, std, p05, p95, min, max.
- Raw samples are written to `timing_samples.csv`.
- Per-view medians are written to `per_view_summary.csv`.
- Sample distributions are plotted in `timing_histograms.svg`.
- Histogram panels are grouped by `(backend, mode, view_index)` and use per-panel bin ranges, so multi-view variation is not mistaken for GPU contention.
- Environment evidence is written into the output directory: `nvidia_smi_*`, `clock_temperature_*`, optional `pmon_during_run.log`.
- `gsplat_native19` is guarded: the script checks that the installed `gsplat.cuda._wrapper.rasterize_to_pixels` supported-channel list includes `19`, otherwise it aborts instead of silently measuring padded `<32u>`.
- `metadata.json` records `gsplat_file`, `gsplat_version`, `gsplat_wrapper_file`, and `native19_supported`.
- Interleaved case order is shuffled with a fixed seed (`--shuffle_seed 0` by default).

Profiler output should only be used for kernel attribution, not horizontal timing comparisons. The `forward` mode in these reports is training-mode forward with autograd graph construction enabled; it is not viewer/inference `torch.no_grad()` timing.

## gsplat Native 19 Patch

Remote gsplat source: `/data2/cse12311966/projects/gsplat-lib`, tag `v1.5.3`.

Tracked patch copy in this repo:

- `profiling_results/gsplat_native19_remote.patch`

Patch contents:

- `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`: add `__INS__(19)`
- `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`: add `__INS__(19)`
- `gsplat/cuda/csrc/Rasterization.cpp`: add `__LAUNCH_KERNEL__(19)` to 3DGS fwd/bwd dispatch
- `gsplat/cuda/_wrapper.py`: add `19` to `rasterize_to_pixels` supported channel list

Confirmed kernel names in profiler reports:

- `rasterize_to_pixels_3dgs_fwd_kernel<19u, float>`
- `rasterize_to_pixels_3dgs_bwd_kernel<19u, float>`

Numerical check on r2 view 0:

- RGB max diff vs shared split-raster: `0.0`
- object max diff vs shared split-raster: `0.0`
- radii equal: `True`

## Invalidated Timing Data

The earlier reports showed cross-run swings larger than the backend differences being discussed. For example, r2 `gsplat_two_pass` forward+backward changed from `9.397 ms` to `15.210 ms` across runs, and shared geometry also varied widely.

Those numbers are useful only as evidence that the old measurement protocol was too noisy.

## New Timing-Only Runs

New protocol output directories:

- `profiling_results/figurines_r2_native19_timing_v2`
- `profiling_results/figurines_r1_native19_timing_v2`

These runs used the fixed interleaved CUDA-event timing path, but they are still not valid for final ranking because the GPU was not idle.

Observed during `nvidia-smi pmon` sampling:

- GPU 0 had the profiling process.
- GPU 0 also had another long-running python process, PID `1981770`, using about `7762 MB` and substantial SM activity.
- GPUs 1-3 also had active python jobs.

So the v2 timing data still reflects GPU contention. It should not be used to make a final backend claim.

Representative v2 medians under contention:

| Dataset | Backend | Forward+backward median ms | Std ms | Status |
| --- | --- | ---: | ---: | --- |
| r2 | `gsplat_two_pass` | 14.834 | 6.560 | contaminated |
| r2 | `gsplat_native19` | 9.527 | 4.797 | contaminated |
| r2 | `gsplat_single` | 11.939 | 5.755 | contaminated |
| r1 | `gsplat_two_pass` | 12.202 | 6.280 | contaminated |
| r1 | `gsplat_native19` | 12.189 | 5.592 | contaminated |
| r1 | `gsplat_single` | 12.283 | 5.809 | contaminated |

The high standard deviations and confirmed GPU contention mean these numbers are not publication-quality.

## Clean Timing Data

Clean protocol:

- GPU: physical GPU 5, selected because it was idle before the run.
- Timing: interleaved CUDA events, no torch profiler on the timing path.
- Views: 10 train views, indices `[0, 17, 34, 51, 68, 85, 102, 119, 136, 153]`.
- Samples: 20 measured samples per view, 200 total samples per backend/mode, after 50 warmup rounds.
- Backends: `diff`, `gsplat_two_pass`, `gsplat_native19`, `gsplat_single`.
- Modes: `forward`, `forward_backward`.
- Native 19 guard: passed, `native19_supported: true`.

Clean output directories:

- `profiling_results/figurines_r2_native19_timing_clean`
- `profiling_results/figurines_r1_native19_timing_clean`

Environment evidence:

- r2 `pmon_during_run.log` shows only PID `2049305` on GPU 5 during the run.
- r1 `pmon_during_run.log` shows only PID `2050136` on GPU 5 during the run.
- Target GPU temperature stayed modest in `nvidia_smi_*` snapshots: r2 GPU 5 from about `41 C` to `49 C`; r1 GPU 5 from about `47 C` to `53 C`.
- `metadata.json` records gsplat path `/data2/cse12311966/projects/gsplat-lib/gsplat/__init__.py`, version `1.5.3`, wrapper `/data2/cse12311966/projects/gsplat-lib/gsplat/cuda/_wrapper.py`, and `native19_supported: true`.

### r2 Clean Summary

Model: `output/lerf/figurines`, resolution 2, image `493x364`, Gaussians `280344`.

| Backend | Mode | Samples | Median ms | Mean ms | Std ms | P05 ms | P95 ms | Peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `diff` | forward | 200 | 2.453 | 2.376 | 0.238 | 1.818 | 2.623 | 229.6 |
| `diff` | forward_backward | 200 | 12.949 | 12.866 | 0.420 | 11.747 | 13.291 | 341.7 |
| `gsplat_two_pass` | forward | 200 | 3.012 | 2.941 | 0.261 | 2.391 | 3.258 | 213.9 |
| `gsplat_two_pass` | forward_backward | 200 | 9.192 | 9.058 | 0.815 | 7.399 | 10.133 | 276.9 |
| `gsplat_native19` | forward | 200 | 2.335 | 2.300 | 0.155 | 2.005 | 2.493 | 217.8 |
| `gsplat_native19` | forward_backward | 200 | 6.511 | 6.457 | 0.535 | 5.364 | 7.233 | 264.8 |
| `gsplat_single` | forward | 200 | 2.707 | 2.643 | 0.247 | 2.108 | 2.940 | 198.5 |
| `gsplat_single` | forward_backward | 200 | 7.743 | 7.609 | 0.824 | 5.867 | 8.663 | 263.8 |

r2 forward+backward ranking:

| Comparison | Median delta | Time reduction |
| --- | ---: | ---: |
| native19 vs two-pass | `-2.682 ms` | `29.2%` |
| native19 vs shared split-raster | `-1.232 ms` | `15.9%` |
| shared split-raster vs two-pass | `-1.449 ms` | `15.8%` |

Per-view result: `gsplat_native19` beats both `gsplat_two_pass` and `gsplat_single` on all 10/10 selected views for forward+backward.

### r1 Clean Summary

Model: `output/lerf/figurines_gsplat_r1_hybrid`, resolution 1, image `986x728`, Gaussians `15095`.

| Backend | Mode | Samples | Median ms | Mean ms | Std ms | P05 ms | P95 ms | Peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `diff` | forward | 200 | 1.951 | 2.002 | 0.137 | 1.859 | 2.268 | 115.4 |
| `diff` | forward_backward | 200 | 44.052 | 45.825 | 5.640 | 40.470 | 61.277 | 172.7 |
| `gsplat_two_pass` | forward | 200 | 2.805 | 2.850 | 0.147 | 2.645 | 3.099 | 95.2 |
| `gsplat_two_pass` | forward_backward | 200 | 7.545 | 7.582 | 0.406 | 6.839 | 8.359 | 181.7 |
| `gsplat_native19` | forward | 200 | 2.386 | 2.410 | 0.102 | 2.283 | 2.586 | 82.4 |
| `gsplat_native19` | forward_backward | 200 | 6.915 | 6.938 | 0.296 | 6.415 | 7.538 | 190.4 |
| `gsplat_single` | forward | 200 | 2.422 | 2.440 | 0.096 | 2.296 | 2.605 | 88.3 |
| `gsplat_single` | forward_backward | 200 | 7.067 | 7.062 | 0.367 | 6.338 | 7.761 | 180.1 |

r1 forward+backward ranking:

| Comparison | Median delta | Time reduction |
| --- | ---: | ---: |
| native19 vs two-pass | `-0.630 ms` | `8.3%` |
| native19 vs shared split-raster | `-0.152 ms` | `2.1%` |
| shared split-raster vs two-pass | `-0.478 ms` | `6.3%` |

Per-view result: `gsplat_native19` beats `gsplat_two_pass` on 10/10 selected views, and beats `gsplat_single` on 9/10 selected views for forward+backward. The one shared-geometry win is view `153`, where shared is faster by `0.071 ms`; this is small relative to the r1 backend margins and should be described as a near tie rather than a large workload-dependent reversal.

The `diff` r1 forward+backward row has a much wider distribution than the gsplat rows (`std 5.640 ms`), so it should be treated as reference context rather than a precise comparison target for the native19 vs shared/two-pass gsplat conclusion.

## Clean Rerun Command

First pick a physically idle GPU. Do not hard-code GPU 0. Example:

```bash
ssh cse12311966@172.18.35.215 'nvidia-smi'
```

Choose a GPU with no compute process and 0% utilization. Suppose GPU 4 is idle:

```bash
ssh cse12311966@172.18.35.215 'cd /data2/cse12311966/projects/gaussian-grouping && source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs && CUDA_VISIBLE_DEVICES=4 python script/profile_render_backends.py -m output/lerf/figurines --backends diff gsplat_two_pass gsplat_native19 gsplat_single --modes forward forward_backward --num_views 10 --view_stride 17 --warmup 50 --iters 20 --memory_iters 3 --gpu_id 4 --monitor_gpu --pmon_samples 999 --out_dir profiling_results/figurines_r2_native19_timing_clean'
```

For r1:

```bash
ssh cse12311966@172.18.35.215 'cd /data2/cse12311966/projects/gaussian-grouping && source ~/miniconda3/etc/profile.d/conda.sh && conda activate gs && CUDA_VISIBLE_DEVICES=4 python script/profile_render_backends.py -m output/lerf/figurines_gsplat_r1_hybrid --backends diff gsplat_two_pass gsplat_native19 gsplat_single --modes forward forward_backward --num_views 10 --view_stride 17 --warmup 50 --iters 20 --memory_iters 3 --gpu_id 4 --monitor_gpu --pmon_samples 999 --out_dir profiling_results/figurines_r1_native19_timing_clean'
```

Each clean output directory should contain:

- `pmon_during_run.log`: target GPU should show only the profiling PID during the run.
- `clock_temperature_before.txt` and `clock_temperature_after.txt`: check for obvious clock drops or thermal throttling.
- `timing_samples.csv`: raw samples for distribution analysis.
- `per_view_summary.csv`: view-by-view median/std, useful for identifying workload-dependent behavior.
- `timing_histograms.svg`: clean runs should look narrow and mostly single-peaked. A broad or bimodal distribution means rerun.

Optional separate monitor:

```bash
ssh cse12311966@172.18.35.215 'nvidia-smi pmon -s um -c 120'
```

Final report should be based on view-wise medians and distribution width, not mean alone. If confidence bands overlap or std remains comparable to the backend gap, report the result as inconclusive.

## Current Defensible Conclusion

Native 19-channel gsplat support is implemented, guarded by the profiling script, functionally correct, and cleanly faster than the previous gsplat alternatives on the tested figurines workloads.

For r2, native19 is the clear best backend: `6.511 ms` forward+backward median, `29.2%` faster than two-pass and `15.9%` faster than shared split-raster, with wins on all 10 selected views.

For r1, native19 is still the best overall but the native19-vs-shared margin is small: `6.915 ms` vs `7.067 ms`, a `2.1%` median time reduction, with native19 winning 9/10 selected views. The safer phrasing is: native19 removes the padding penalty and is the preferred implementation in these clean tests, while shared split-raster remains close on the small-Gaussian r1 workload.

Production configuration follows that tradeoff explicitly: the default gsplat path remains shared-geometry split-raster so the repository works with stock gsplat, while the clean-run winner can be enabled with `USE_GSPLAT_NATIVE19=1`. That opt-in path performs a native19 capability check at import time and raises a `RuntimeError` if the installed gsplat wrapper does not advertise 19-channel support, avoiding silent fallback to padded `<32u>`.
