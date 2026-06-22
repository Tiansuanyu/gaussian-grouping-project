# Render Backend Profiling

- model_path: `output/lerf/ramen`
- iteration: `30000`
- view_index: `0`
- selected views: `[0, 17, 34, 51, 68, 85, 102, 119, 5, 22]`
- image: `494x366`
- gaussians: `1114069`
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
| diff | forward | 200 | 5.848 | 5.960 | 0.567 | 5.090 | 5.217 | 7.084 | 7.160 | 1156.3 |  |
| diff | forward_backward | 200 | 22.539 | 22.915 | 1.166 | 21.104 | 21.290 | 24.846 | 25.112 | 1561.9 |  |
| gsplat_two_pass | forward | 200 | 6.778 | 7.032 | 0.981 | 5.749 | 5.796 | 8.668 | 8.837 | 1123.5 |  |
| gsplat_two_pass | forward_backward | 200 | 24.852 | 25.275 | 3.067 | 21.247 | 21.414 | 30.562 | 30.948 | 1382.6 |  |
| gsplat_native19 | forward | 200 | 5.476 | 5.476 | 0.532 | 4.776 | 4.816 | 6.373 | 6.648 | 1146.7 |  |
| gsplat_native19 | forward_backward | 200 | 16.751 | 16.912 | 1.898 | 14.436 | 14.535 | 20.094 | 20.509 | 1334.4 |  |
| gsplat_single | forward | 200 | 5.989 | 6.213 | 0.948 | 4.969 | 5.028 | 7.786 | 8.010 | 1066.6 |  |
| gsplat_single | forward_backward | 200 | 19.652 | 20.078 | 3.023 | 16.130 | 16.297 | 25.335 | 25.871 | 1335.4 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.