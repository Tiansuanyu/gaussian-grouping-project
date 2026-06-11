# Render Backend Profiling

- model_path: `output/lerf/figurines_gsplat_r1_hybrid`
- iteration: `30000`
- view_index: `0`
- image: `986x728`
- gaussians: `15095`
- timed iterations: `10` after `3` warmup
- profiler iterations: `5` after `3` warmup
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| gsplat_two_pass | forward | 3.090 | 3.064 | 3.132 | 90.2 | 90.2 |
| gsplat_two_pass | forward_backward | 9.124 | 8.537 | 9.339 | 180.4 | 180.4 |
| gsplat_padded_single | forward | 3.438 | 3.407 | 3.486 | 118.6 | 118.6 |
| gsplat_padded_single | forward_backward | 11.327 | 11.252 | 11.353 | 254.3 | 254.3 |
| gsplat_single | forward | 2.707 | 2.696 | 2.721 | 86.1 | 86.1 |
| gsplat_single | forward_backward | 7.350 | 7.309 | 7.379 | 179.3 | 179.3 |

## Top CUDA Rows

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 5.896 | 0.5896 | 5.895 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.911 | 0.7822 | 3.911 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.984 | 0.3968 | 1.984 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.636 | 0.0273 | 1.636 |
| 5 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.174 | 0.0587 | 1.174 |
| 6 | aten::mean | 10 | 0.561 | 0.0561 | 0.552 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.542 | 0.0542 | 0.542 |
| 8 | cudaMemsetAsync | 90 | 0.419 | 0.0047 | 0.419 |
| 9 | aten::linalg_inv | 5 | 0.321 | 0.0642 | 0.000 |
| 10 | aten::linalg_inv_ex | 5 | 0.279 | 0.0558 | 0.000 |
| 11 | aten::inverse | 5 | 0.258 | 0.0516 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.249 | 0.0498 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.249 | 0.0498 | 0.000 |
| 14 | Memset (Device) | 90 | 0.231 | 0.0026 | 0.231 |
| 15 | cudaLaunchKernel | 305 | 0.214 | 0.0007 | 0.214 |
| 16 | aten::cumsum | 10 | 0.196 | 0.0196 | 0.085 |
| 17 | aten::linalg_lu_solve | 5 | 0.151 | 0.0302 | 0.129 |
| 18 | aten::copy_ | 30 | 0.131 | 0.0044 | 0.106 |
| 19 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 10 | 0.131 | 0.0131 | 0.131 |
| 20 | aten::gt | 10 | 0.118 | 0.0118 | 0.031 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 20.771 | 2.0771 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 20.771 | 2.0771 | 20.603 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 13.906 | 2.7812 | 13.906 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 6.697 | 1.3394 | 6.697 |
| 5 | _RasterizeToPixels | 10 | 5.469 | 0.5469 | 5.469 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.642 | 0.7284 | 3.642 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.827 | 0.3654 | 1.827 |
| 8 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.566 | 0.0261 | 1.566 |
| 9 | aten::copy_ | 95 | 1.562 | 0.0164 | 1.562 |
| 10 | autograd::engine::evaluate_function: SelectBackward0 | 10 | 1.478 | 0.1478 | 0.000 |
| 11 | SelectBackward0 | 10 | 1.478 | 0.1478 | 0.000 |
| 12 | aten::select_backward | 10 | 1.478 | 0.1478 | 0.000 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.119 | 0.0559 | 1.119 |
| 14 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.095 | 0.0548 | 1.095 |
| 15 | aten::fill_ | 200 | 1.029 | 0.0051 | 1.029 |
| 16 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 0.989 | 0.0053 | 0.989 |
| 17 | aten::zero_ | 180 | 0.972 | 0.0054 | 0.000 |
| 18 | aten::zeros | 95 | 0.742 | 0.0078 | 0.000 |
| 19 | aten::div | 35 | 0.590 | 0.0169 | 0.590 |
| 20 | aten::mean | 10 | 0.547 | 0.0547 | 0.547 |

### gsplat_padded_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 7.313 | 1.4626 | 7.313 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 7.313 | 1.4626 | 7.313 |
| 3 | aten::mean | 10 | 1.183 | 0.1183 | 1.183 |
| 4 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 1.173 | 0.1173 | 1.173 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.782 | 0.0261 | 0.782 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.540 | 0.0540 | 0.540 |
| 7 | aten::inverse | 5 | 0.296 | 0.0592 | 0.000 |
| 8 | aten::linalg_inv | 5 | 0.296 | 0.0592 | 0.000 |
| 9 | aten::linalg_inv_ex | 5 | 0.256 | 0.0512 | 0.000 |
| 10 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 11 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 12 | aten::cat | 25 | 0.198 | 0.0079 | 0.198 |
| 13 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 0.158 | 0.0105 | 0.158 |
| 14 | aten::linalg_lu_solve | 5 | 0.142 | 0.0284 | 0.120 |
| 15 | Memset (Device) | 50 | 0.118 | 0.0024 | 0.118 |
| 16 | aten::linalg_lu_factor_ex | 5 | 0.089 | 0.0178 | 0.074 |
| 17 | aten::copy_ | 25 | 0.086 | 0.0034 | 0.086 |
| 18 | aten::fill_ | 30 | 0.076 | 0.0025 | 0.076 |
| 19 | aten::cumsum | 5 | 0.075 | 0.0150 | 0.045 |
| 20 | aten::zero_ | 25 | 0.061 | 0.0024 | 0.000 |

### gsplat_padded_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 30.851 | 6.1702 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 30.851 | 6.1702 | 30.765 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 30.765 | 6.1530 | 30.765 |
| 4 | _RasterizeToPixels | 5 | 7.244 | 1.4488 | 7.244 |
| 5 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 7.244 | 1.4488 | 7.244 |
| 6 | autograd::engine::evaluate_function: SliceBackward0 | 25 | 5.622 | 0.2249 | 0.000 |
| 7 | SliceBackward0 | 25 | 4.286 | 0.1714 | 0.000 |
| 8 | aten::slice_backward | 25 | 4.286 | 0.1714 | 0.000 |
| 9 | aten::copy_ | 70 | 3.828 | 0.0547 | 3.828 |
| 10 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 2.770 | 0.0923 | 2.770 |
| 11 | aten::fill_ | 140 | 2.339 | 0.0167 | 2.339 |
| 12 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 2.309 | 0.0178 | 2.309 |
| 13 | aten::zero_ | 120 | 2.288 | 0.0191 | 0.000 |
| 14 | aten::zeros | 70 | 2.146 | 0.0307 | 0.000 |
| 15 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3>) | 20 | 1.381 | 0.0691 | 1.381 |
| 16 | autograd::engine::evaluate_function: SelectBackward0 | 5 | 1.378 | 0.2756 | 0.000 |
| 17 | SelectBackward0 | 5 | 1.378 | 0.2756 | 0.000 |
| 18 | aten::select_backward | 5 | 1.378 | 0.2756 | 0.000 |
| 19 | aten::add_ | 10 | 1.351 | 0.1351 | 1.351 |
| 20 | aten::mean | 10 | 1.184 | 0.1184 | 1.184 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 5.278 | 0.5278 | 5.278 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.506 | 0.7012 | 3.506 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.772 | 0.3544 | 1.772 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.776 | 0.0259 | 0.776 |
| 5 | aten::mean | 10 | 0.540 | 0.0540 | 0.540 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.530 | 0.0530 | 0.530 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.530 | 0.0530 | 0.530 |
| 8 | aten::linalg_inv | 5 | 0.292 | 0.0584 | 0.000 |
| 9 | aten::linalg_inv_ex | 5 | 0.252 | 0.0504 | 0.000 |
| 10 | aten::linalg_solve_ex | 5 | 0.227 | 0.0454 | 0.000 |
| 11 | aten::_linalg_solve_ex | 5 | 0.227 | 0.0454 | 0.000 |
| 12 | aten::inverse | 5 | 0.175 | 0.0350 | 0.000 |
| 13 | aten::linalg_lu_solve | 5 | 0.139 | 0.0278 | 0.119 |
| 14 | Memset (Device) | 50 | 0.116 | 0.0023 | 0.116 |
| 15 | aten::linalg_lu_factor_ex | 5 | 0.088 | 0.0176 | 0.073 |
| 16 | aten::copy_ | 25 | 0.085 | 0.0034 | 0.085 |
| 17 | aten::cumsum | 5 | 0.077 | 0.0154 | 0.047 |
| 18 | aten::cat | 5 | 0.071 | 0.0142 | 0.071 |
| 19 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 0.071 | 0.0142 | 0.071 |
| 20 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 5 | 0.063 | 0.0126 | 0.063 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 20.308 | 2.0308 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 20.262 | 2.0262 | 20.105 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 13.541 | 2.7082 | 13.541 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 6.564 | 1.3128 | 6.564 |
| 5 | _RasterizeToPixels | 10 | 5.264 | 0.5264 | 5.264 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.486 | 0.6972 | 3.486 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.778 | 0.3556 | 1.778 |
| 8 | autograd::engine::evaluate_function: SelectBackward0 | 10 | 1.480 | 0.1480 | 0.000 |
| 9 | SelectBackward0 | 10 | 1.480 | 0.1480 | 0.000 |
| 10 | aten::select_backward | 10 | 1.480 | 0.1480 | 0.000 |
| 11 | aten::copy_ | 55 | 1.236 | 0.0225 | 1.236 |
| 12 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.117 | 0.0559 | 1.117 |
| 13 | aten::fill_ | 140 | 0.802 | 0.0057 | 0.802 |
| 14 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.788 | 0.0263 | 0.788 |
| 15 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 0.772 | 0.0059 | 0.772 |
| 16 | aten::zero_ | 120 | 0.754 | 0.0063 | 0.000 |
| 17 | aten::div | 35 | 0.580 | 0.0166 | 0.580 |
| 18 | aten::zeros | 50 | 0.577 | 0.0115 | 0.000 |
| 19 | aten::mean | 10 | 0.546 | 0.0546 | 0.546 |
| 20 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.536 | 0.0536 | 0.536 |