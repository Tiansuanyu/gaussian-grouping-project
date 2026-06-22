# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- image: `493x364`
- gaussians: `280344`
- timed iterations: `1` after `1` warmup
- profiler iterations: `1` after `1` warmup

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| diff | forward | 4.821 | 4.821 | 4.821 | 226.8 | 226.8 |
| gsplat_two_pass | forward | 11.763 | 11.763 | 11.763 | 213.9 | 213.9 |
| gsplat_single | forward | 9.436 | 9.436 | 9.436 | 260.7 | 260.7 |

## Top CUDA Rows

### diff_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeGaussians | 1 | 1.847 | 1.8470 | 1.811 |
| 2 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 1 | 1.237 | 1.2370 | 1.237 |
| 3 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 6 | 0.286 | 0.0477 | 0.286 |
| 4 | aten::cat | 1 | 0.277 | 0.2770 | 0.277 |
| 5 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 1 | 0.277 | 0.2770 | 0.277 |

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 2 | 1.251 | 0.6255 | 1.251 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 1 | 0.691 | 0.6910 | 0.691 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 1 | 0.560 | 0.5600 | 0.560 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 12 | 0.327 | 0.0272 | 0.327 |
| 5 | aten::cat | 1 | 0.279 | 0.2790 | 0.279 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 1 | 0.994 | 0.9940 | 0.994 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 1 | 0.994 | 0.9940 | 0.994 |
| 3 | aten::cat | 5 | 0.568 | 0.1136 | 0.568 |
| 4 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 3 | 0.560 | 0.1867 | 0.560 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 6 | 0.163 | 0.0272 | 0.163 |