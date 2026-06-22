# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- image: `493x364`
- gaussians: `280344`
- timing protocol: interleaved CUDA events, profiler disabled
- timed samples per backend/mode: `200` after `50` warmup rounds
- profiler pass: `disabled`
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Samples | Median ms | Mean ms | Std ms | Min ms | P05 ms | P95 ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsplat_two_pass | forward | 200 | 4.175 | 6.788 | 3.854 | 2.943 | 3.605 | 12.431 | 12.820 | 214.2 |  |
| gsplat_two_pass | forward_backward | 200 | 14.834 | 17.779 | 6.560 | 9.106 | 11.303 | 26.896 | 29.563 | 276.6 |  |
| gsplat_native19 | forward | 200 | 3.032 | 5.071 | 2.939 | 2.265 | 2.562 | 9.209 | 9.504 | 217.3 |  |
| gsplat_native19 | forward_backward | 200 | 9.527 | 11.988 | 4.797 | 6.547 | 7.533 | 18.810 | 19.595 | 264.9 |  |
| gsplat_single | forward | 200 | 3.254 | 5.358 | 2.922 | 2.614 | 2.892 | 9.512 | 9.733 | 198.9 |  |
| gsplat_single | forward_backward | 200 | 11.939 | 14.783 | 5.755 | 7.708 | 9.275 | 22.921 | 23.209 | 264.9 |  |

## Top CUDA Rows

Profiler pass disabled. Use `--profile` for kernel breakdown; do not use profiler timings for backend ranking.