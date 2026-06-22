# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- image: `493x364`
- gaussians: `280344`
- timed iterations: `30` after `5` warmup
- profiler iterations: `5` after `5` warmup
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| gsplat_two_pass | forward | 3.819 | 3.031 | 5.531 | 214.8 | 214.8 |
| gsplat_two_pass | forward_backward | 15.210 | 9.446 | 26.820 | 276.6 | 276.6 |
| gsplat_padded_single | forward | 4.848 | 2.559 | 9.262 | 217.3 | 217.3 |
| gsplat_padded_single | forward_backward | 10.888 | 7.001 | 18.748 | 264.9 | 264.9 |
| gsplat_single | forward | 3.260 | 2.870 | 3.980 | 198.9 | 198.9 |
| gsplat_single | forward_backward | 13.855 | 8.176 | 23.036 | 264.9 | 264.9 |

## Top CUDA Rows

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.292 | 0.6292 | 6.291 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.500 | 0.7000 | 3.500 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.791 | 0.5582 | 2.791 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.624 | 0.0271 | 1.624 |
| 5 | aten::cat | 5 | 1.389 | 0.2778 | 1.386 |
| 6 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.386 | 0.2772 | 1.386 |
| 7 | cudaLaunchKernel | 305 | 0.776 | 0.0025 | 0.776 |
| 8 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 0.668 | 0.0334 | 0.668 |
| 9 | cudaMemsetAsync | 90 | 0.661 | 0.0073 | 0.661 |
| 10 | _FullyFusedProjection | 10 | 0.437 | 0.0437 | 0.432 |
| 11 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 10 | 0.432 | 0.0432 | 0.432 |
| 12 | aten::exp | 5 | 0.371 | 0.0742 | 0.071 |
| 13 | aten::inverse | 5 | 0.319 | 0.0638 | 0.000 |
| 14 | aten::linalg_inv | 5 | 0.319 | 0.0638 | 0.000 |
| 15 | aten::cumsum | 10 | 0.290 | 0.0290 | 0.141 |
| 16 | aten::linalg_inv_ex | 5 | 0.269 | 0.0538 | 0.000 |
| 17 | aten::mean | 10 | 0.265 | 0.0265 | 0.221 |
| 18 | _SphericalHarmonics | 5 | 0.241 | 0.0482 | 0.218 |
| 19 | aten::linalg_solve_ex | 5 | 0.238 | 0.0476 | 0.000 |
| 20 | aten::_linalg_solve_ex | 5 | 0.238 | 0.0476 | 0.000 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 26.774 | 2.6774 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 26.774 | 2.6774 | 26.385 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 17.116 | 3.4232 | 17.116 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 9.269 | 1.8538 | 9.269 |
| 5 | autograd::engine::evaluate_function: SliceBackward0 | 45 | 8.468 | 0.1882 | 0.000 |
| 6 | SliceBackward0 | 45 | 8.468 | 0.1882 | 0.000 |
| 7 | aten::slice_backward | 45 | 8.468 | 0.1882 | 0.000 |
| 8 | _RasterizeToPixels | 10 | 6.308 | 0.6308 | 6.308 |
| 9 | aten::fill_ | 200 | 6.146 | 0.0307 | 6.146 |
| 10 | aten::zero_ | 180 | 6.088 | 0.0338 | 0.000 |
| 11 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 6.083 | 0.0329 | 6.083 |
| 12 | aten::copy_ | 95 | 5.276 | 0.0555 | 5.276 |
| 13 | aten::zeros | 95 | 5.013 | 0.0528 | 0.000 |
| 14 | autograd::engine::evaluate_function: _FullyFusedProjectionBackward | 10 | 4.335 | 0.4335 | 0.000 |
| 15 | _FullyFusedProjectionBackward | 10 | 3.924 | 0.3924 | 3.602 |
| 16 | Memcpy DtoD (Device -> Device) | 55 | 3.772 | 0.0686 | 3.772 |
| 17 | void gsplat::projection_ewa_3dgs_fused_bwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, gsplat::CameraModelType, int const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*) | 10 | 3.602 | 0.3602 | 3.602 |
| 18 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.507 | 0.7014 | 3.507 |
| 19 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.801 | 0.5602 | 2.801 |
| 20 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.607 | 0.0268 | 1.607 |

### gsplat_padded_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 3.922 | 0.7844 | 3.922 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.922 | 0.7844 | 3.922 |
| 3 | aten::cat | 15 | 1.983 | 0.1322 | 1.983 |
| 4 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 10 | 1.963 | 0.1963 | 1.963 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.812 | 0.0271 | 0.812 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.344 | 0.0344 | 0.344 |
| 7 | aten::mean | 10 | 0.329 | 0.0329 | 0.329 |
| 8 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.319 | 0.0319 | 0.319 |
| 9 | aten::inverse | 5 | 0.304 | 0.0608 | 0.000 |
| 10 | aten::linalg_inv | 5 | 0.304 | 0.0608 | 0.000 |
| 11 | aten::linalg_inv_ex | 5 | 0.261 | 0.0522 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 14 | _FullyFusedProjection | 5 | 0.222 | 0.0444 | 0.222 |
| 15 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.222 | 0.0444 | 0.222 |
| 16 | _SphericalHarmonics | 5 | 0.222 | 0.0444 | 0.222 |
| 17 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.222 | 0.0444 | 0.222 |
| 18 | aten::linalg_lu_solve | 5 | 0.142 | 0.0284 | 0.120 |
| 19 | Memset (Device) | 50 | 0.128 | 0.0026 | 0.128 |
| 20 | aten::linalg_vector_norm | 5 | 0.120 | 0.0240 | 0.120 |

### gsplat_padded_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 34.523 | 6.9046 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 34.523 | 6.9046 | 34.248 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 34.248 | 6.8496 | 34.248 |
| 4 | aten::copy_ | 65 | 5.236 | 0.0806 | 5.236 |
| 5 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 25 | 4.678 | 0.1871 | 4.678 |
| 6 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 4.267 | 0.1219 | 0.000 |
| 7 | torch::autograd::AccumulateGrad | 35 | 4.267 | 0.1219 | 0.000 |
| 8 | _RasterizeToPixels | 5 | 3.878 | 0.7756 | 3.878 |
| 9 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.878 | 0.7756 | 3.878 |
| 10 | aten::cat | 20 | 2.063 | 0.1032 | 2.063 |
| 11 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 10 | 1.962 | 0.1962 | 1.962 |
| 12 | aten::fill_ | 125 | 1.460 | 0.0117 | 1.460 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 115 | 1.420 | 0.0123 | 1.420 |
| 14 | aten::zero_ | 105 | 1.403 | 0.0134 | 0.000 |
| 15 | autograd::engine::evaluate_function: SliceBackward0 | 20 | 1.240 | 0.0620 | 0.000 |
| 16 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 1.063 | 0.2126 | 0.000 |
| 17 | _SphericalHarmonicsBackward | 5 | 1.063 | 0.2126 | 0.606 |
| 18 | SliceBackward0 | 20 | 0.892 | 0.0446 | 0.000 |
| 19 | aten::slice_backward | 20 | 0.892 | 0.0446 | 0.000 |
| 20 | aten::zeros_like | 45 | 0.837 | 0.0186 | 0.000 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.258 | 0.6258 | 6.258 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.502 | 0.7004 | 3.502 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.756 | 0.5512 | 2.756 |
| 4 | aten::cat | 5 | 1.374 | 0.2748 | 1.374 |
| 5 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.374 | 0.2748 | 1.374 |
| 6 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.808 | 0.0269 | 0.808 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.354 | 0.0354 | 0.354 |
| 8 | aten::linalg_inv | 5 | 0.308 | 0.0616 | 0.000 |
| 9 | aten::linalg_inv_ex | 5 | 0.264 | 0.0528 | 0.000 |
| 10 | aten::inverse | 5 | 0.245 | 0.0490 | 0.000 |
| 11 | aten::linalg_solve_ex | 5 | 0.234 | 0.0468 | 0.000 |
| 12 | aten::_linalg_solve_ex | 5 | 0.234 | 0.0468 | 0.000 |
| 13 | _FullyFusedProjection | 5 | 0.221 | 0.0442 | 0.221 |
| 14 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.221 | 0.0442 | 0.221 |
| 15 | _SphericalHarmonics | 5 | 0.221 | 0.0442 | 0.221 |
| 16 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.221 | 0.0442 | 0.221 |
| 17 | aten::mean | 10 | 0.221 | 0.0221 | 0.221 |
| 18 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.211 | 0.0211 | 0.211 |
| 19 | aten::linalg_lu_solve | 5 | 0.145 | 0.0290 | 0.122 |
| 20 | aten::linalg_vector_norm | 5 | 0.123 | 0.0246 | 0.123 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 21.414 | 2.1414 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 21.187 | 2.1187 | 20.802 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 13.011 | 2.6022 | 13.011 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 7.791 | 1.5582 | 7.791 |
| 5 | _RasterizeToPixels | 10 | 6.174 | 0.6174 | 6.174 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.443 | 0.6886 | 3.443 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.731 | 0.5462 | 2.731 |
| 8 | aten::copy_ | 55 | 1.653 | 0.0301 | 1.653 |
| 9 | aten::cat | 10 | 1.443 | 0.1443 | 1.443 |
| 10 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.382 | 0.0691 | 1.382 |
| 11 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.363 | 0.2726 | 1.363 |
| 12 | aten::fill_ | 140 | 1.348 | 0.0096 | 1.348 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 1.308 | 0.0101 | 1.308 |
| 14 | aten::zero_ | 120 | 1.280 | 0.0107 | 0.000 |
| 15 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 1.093 | 0.0312 | 0.000 |
| 16 | torch::autograd::AccumulateGrad | 35 | 1.093 | 0.0312 | 0.000 |
| 17 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 1.072 | 0.2144 | 0.000 |
| 18 | _SphericalHarmonicsBackward | 5 | 1.072 | 0.2144 | 0.602 |
| 19 | aten::zeros_like | 65 | 0.941 | 0.0145 | 0.000 |
| 20 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.798 | 0.0266 | 0.798 |