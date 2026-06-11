# Render Backend Profiling

- model_path: `output/lerf/figurines_gsplat_r1_hybrid`
- iteration: `30000`
- view_index: `0`
- image: `986x728`
- gaussians: `15095`
- timing protocol: interleaved CUDA events, profiler disabled
- timed samples per backend/mode: `200` after `50` warmup rounds
- profiler pass: `disabled`
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Samples | Median ms | Mean ms | Std ms | Min ms | P05 ms | P95 ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsplat_two_pass | forward | 200 | 4.120 | 6.995 | 3.871 | 2.793 | 3.236 | 11.725 | 11.993 | 90.2 |  |
| gsplat_two_pass | forward_backward | 200 | 12.202 | 15.243 | 6.280 | 7.505 | 8.786 | 24.377 | 25.154 | 180.4 |  |
| gsplat_native19 | forward | 200 | 3.017 | 5.108 | 2.849 | 2.400 | 2.552 | 8.870 | 9.243 | 80.7 |  |
| gsplat_native19 | forward_backward | 200 | 12.189 | 14.285 | 5.592 | 6.771 | 8.601 | 21.885 | 22.608 | 189.9 |  |
| gsplat_single | forward | 200 | 2.955 | 5.130 | 2.876 | 2.425 | 2.622 | 8.924 | 9.179 | 86.1 |  |
| gsplat_single | forward_backward | 200 | 12.283 | 14.569 | 5.809 | 6.982 | 8.470 | 22.091 | 27.387 | 179.3 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.