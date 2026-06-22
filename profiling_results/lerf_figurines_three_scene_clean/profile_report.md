# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- selected views: `[0, 17, 34, 51, 68, 85, 102, 119, 136, 153]`
- image: `493x364`
- gaussians: `280344`
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
| diff | forward | 200 | 2.429 | 2.341 | 0.236 | 1.764 | 1.789 | 2.584 | 2.639 | 229.6 |  |
| diff | forward_backward | 200 | 12.899 | 12.838 | 0.414 | 11.729 | 11.796 | 13.352 | 13.761 | 341.7 |  |
| gsplat_two_pass | forward | 200 | 2.950 | 2.875 | 0.251 | 2.274 | 2.325 | 3.165 | 3.260 | 213.9 |  |
| gsplat_two_pass | forward_backward | 200 | 9.091 | 8.989 | 0.794 | 7.200 | 7.400 | 10.031 | 10.279 | 276.9 |  |
| gsplat_native19 | forward | 200 | 2.264 | 2.230 | 0.155 | 1.892 | 1.923 | 2.412 | 2.639 | 217.8 |  |
| gsplat_native19 | forward_backward | 200 | 6.445 | 6.404 | 0.509 | 5.196 | 5.380 | 7.135 | 7.341 | 264.8 |  |
| gsplat_single | forward | 200 | 2.635 | 2.584 | 0.237 | 2.028 | 2.061 | 2.871 | 2.931 | 198.5 |  |
| gsplat_single | forward_backward | 200 | 7.695 | 7.535 | 0.814 | 5.733 | 5.839 | 8.607 | 8.727 | 263.8 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.