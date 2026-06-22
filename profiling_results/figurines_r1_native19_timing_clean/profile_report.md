# Render Backend Profiling

- model_path: `output/lerf/figurines_gsplat_r1_hybrid`
- iteration: `30000`
- view_index: `0`
- selected views: `[0, 17, 34, 51, 68, 85, 102, 119, 136, 153]`
- image: `986x728`
- gaussians: `15095`
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
| diff | forward | 200 | 1.951 | 2.002 | 0.137 | 1.809 | 1.859 | 2.268 | 2.437 | 115.4 |  |
| diff | forward_backward | 200 | 44.052 | 45.825 | 5.640 | 40.359 | 40.470 | 61.277 | 61.587 | 172.7 |  |
| gsplat_two_pass | forward | 200 | 2.805 | 2.850 | 0.147 | 2.597 | 2.645 | 3.099 | 3.453 | 95.2 |  |
| gsplat_two_pass | forward_backward | 200 | 7.545 | 7.582 | 0.406 | 6.769 | 6.839 | 8.359 | 8.543 | 181.7 |  |
| gsplat_native19 | forward | 200 | 2.386 | 2.410 | 0.102 | 2.242 | 2.283 | 2.586 | 2.919 | 82.4 |  |
| gsplat_native19 | forward_backward | 200 | 6.915 | 6.938 | 0.296 | 6.337 | 6.415 | 7.538 | 7.626 | 190.4 |  |
| gsplat_single | forward | 200 | 2.422 | 2.440 | 0.096 | 2.257 | 2.296 | 2.605 | 2.770 | 88.3 |  |
| gsplat_single | forward_backward | 200 | 7.067 | 7.062 | 0.367 | 6.309 | 6.338 | 7.761 | 7.795 | 180.1 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.