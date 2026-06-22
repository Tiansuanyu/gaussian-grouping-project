# Three-Scene Clean Profiling Summary

Protocol:

- Backends: `diff`, `gsplat_two_pass`, `gsplat_native19`, `gsplat_single`
- Modes: `forward`, `forward_backward`
- Views: 10 train views per scene, `view_stride=17`
- Timing: interleaved CUDA events, `--warmup 50`, `--iters 20`
- GPU selection: each run used an idle GPU chosen from `nvidia-smi`
- Trust rule: `pmon_during_run.log` showed only the profiling PID on the target GPU for all three runs
- Aggregation: per-view medians first, then summary statistics across views

## Combined Table

| Scene | Gaussians | Backend | Mode | Median ms | Std ms | Peak MB |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| ramen | 1114069 | diff | forward | 5.820 | 0.590 | 1156.3 |
| ramen | 1114069 | diff | forward_backward | 22.415 | 1.218 | 1561.9 |
| ramen | 1114069 | gsplat_two_pass | forward | 6.748 | 1.037 | 1123.5 |
| ramen | 1114069 | gsplat_two_pass | forward_backward | 24.886 | 3.232 | 1382.6 |
| ramen | 1114069 | gsplat_native19 | forward | 5.359 | 0.552 | 1146.7 |
| ramen | 1114069 | gsplat_native19 | forward_backward | 16.722 | 1.984 | 1334.4 |
| ramen | 1114069 | gsplat_single | forward | 5.973 | 0.996 | 1066.6 |
| ramen | 1114069 | gsplat_single | forward_backward | 19.675 | 3.163 | 1335.4 |
| figurines | 280344 | diff | forward | 2.423 | 0.248 | 229.6 |
| figurines | 280344 | diff | forward_backward | 12.901 | 0.427 | 341.7 |
| figurines | 280344 | gsplat_two_pass | forward | 2.935 | 0.259 | 213.9 |
| figurines | 280344 | gsplat_two_pass | forward_backward | 9.101 | 0.819 | 276.9 |
| figurines | 280344 | gsplat_native19 | forward | 2.263 | 0.156 | 217.8 |
| figurines | 280344 | gsplat_native19 | forward_backward | 6.457 | 0.523 | 264.8 |
| figurines | 280344 | gsplat_single | forward | 2.628 | 0.247 | 198.5 |
| figurines | 280344 | gsplat_single | forward_backward | 7.699 | 0.853 | 263.8 |
| teatime | 279705 | diff | forward | 2.109 | 0.294 | 221.6 |
| teatime | 279705 | diff | forward_backward | 11.462 | 0.750 | 335.9 |
| teatime | 279705 | gsplat_two_pass | forward | 2.677 | 0.396 | 213.5 |
| teatime | 279705 | gsplat_two_pass | forward_backward | 8.289 | 1.258 | 276.6 |
| teatime | 279705 | gsplat_native19 | forward | 2.199 | 0.239 | 217.7 |
| teatime | 279705 | gsplat_native19 | forward_backward | 6.019 | 0.811 | 265.0 |
| teatime | 279705 | gsplat_single | forward | 2.393 | 0.401 | 199.0 |
| teatime | 279705 | gsplat_single | forward_backward | 6.803 | 1.316 | 263.7 |

## Takeaway

Native19 wins on all three scenes in the forward+backward setting, and it is also best in forward-only timing. The advantage over shared split-raster is strong on ramen (`14.6%`), figurines (`15.6%`), and still present on teatime (`11.3%`).

That said, the claim that the native19 edge strictly grows monotonically with Gaussian count is only weakly supported by these three points. Ramen is the heaviest scene and does show a clear native19 win, but figurines is slightly stronger than ramen on the paired-median gap, while teatime is smaller. The safe conclusion is:

- native19 is consistently the fastest backend across all three scenes,
- the speedup is more visible on heavier workloads than on the smallest one,
- but the growth-vs-count trend is not perfectly monotonic in this sample.

