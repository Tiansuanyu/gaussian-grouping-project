# Render Backend Profiling

- model_path: `output/lerf/teatime`
- iteration: `30000`
- view_index: `0`
- selected views: `[0, 17, 34, 51, 68, 85, 102, 119, 136, 153]`
- image: `494x365`
- gaussians: `279705`
- timing protocol: interleaved CUDA events, profiler disabled
- forward mode: training-mode forward, autograd graph enabled
- shuffle seed: `0`
- timed samples per backend/mode: `200` (`20` per view) after `50` warmup rounds
- environment logs: `nvidia_smi_before.txt`, `clock_temperature_before.txt`, `nvidia_smi_after.txt`, `clock_temperature_after.txt`
- pmon log: `pmon_during_run.log`
- profiler pass: `disabled`
- chart: [`summary_chart.svg`](summary_chart.svg)
- histograms: [`timing_histograms.svg`](timing_histograms.svg) for modes `['forward_backward']` and first `3` selected views

## Summary

| Backend | Mode | Samples | Median ms | Mean ms | Std ms | Min ms | P05 ms | P95 ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diff | forward | 200 | 2.108 | 2.067 | 0.284 | 1.635 | 1.678 | 2.519 | 2.582 | 221.6 |  |
| diff | forward_backward | 200 | 11.444 | 11.553 | 0.715 | 10.267 | 10.414 | 13.157 | 13.316 | 335.9 |  |
| gsplat_two_pass | forward | 200 | 2.632 | 2.705 | 0.382 | 1.997 | 2.214 | 3.325 | 3.411 | 213.5 |  |
| gsplat_two_pass | forward_backward | 200 | 8.223 | 8.777 | 6.209 | 6.003 | 6.612 | 10.229 | 94.427 | 276.6 |  |
| gsplat_native19 | forward | 200 | 2.192 | 2.199 | 0.243 | 1.727 | 1.876 | 2.573 | 3.059 | 217.7 |  |
| gsplat_native19 | forward_backward | 200 | 5.967 | 6.051 | 0.796 | 4.477 | 4.887 | 7.268 | 7.608 | 265.0 |  |
| gsplat_single | forward | 200 | 2.367 | 2.428 | 0.387 | 1.732 | 1.888 | 3.064 | 3.118 | 199.0 |  |
| gsplat_single | forward_backward | 200 | 6.773 | 6.894 | 1.260 | 4.661 | 5.099 | 8.838 | 8.904 | 263.7 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.