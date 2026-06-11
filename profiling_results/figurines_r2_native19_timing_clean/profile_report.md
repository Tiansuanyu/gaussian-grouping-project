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
| diff | forward | 200 | 2.453 | 2.376 | 0.238 | 1.797 | 1.818 | 2.623 | 2.663 | 229.6 |  |
| diff | forward_backward | 200 | 12.949 | 12.866 | 0.420 | 11.565 | 11.747 | 13.291 | 13.498 | 341.7 |  |
| gsplat_two_pass | forward | 200 | 3.012 | 2.941 | 0.261 | 2.332 | 2.391 | 3.258 | 3.611 | 213.9 |  |
| gsplat_two_pass | forward_backward | 200 | 9.192 | 9.058 | 0.815 | 7.293 | 7.399 | 10.133 | 10.242 | 276.9 |  |
| gsplat_native19 | forward | 200 | 2.335 | 2.300 | 0.155 | 1.935 | 2.005 | 2.493 | 2.657 | 217.8 |  |
| gsplat_native19 | forward_backward | 200 | 6.511 | 6.457 | 0.535 | 5.223 | 5.364 | 7.233 | 7.617 | 264.8 |  |
| gsplat_single | forward | 200 | 2.707 | 2.643 | 0.247 | 2.058 | 2.108 | 2.940 | 3.077 | 198.5 |  |
| gsplat_single | forward_backward | 200 | 7.743 | 7.609 | 0.824 | 5.791 | 5.867 | 8.663 | 9.013 | 263.8 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.