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
| gsplat_two_pass | forward | 4.078 | 2.893 | 6.152 | 90.2 | 90.2 |
| gsplat_two_pass | forward_backward | 12.344 | 8.028 | 20.499 | 180.4 | 180.4 |
| gsplat_single | forward | 8.361 | 3.696 | 9.512 | 118.6 | 118.6 |
| gsplat_single | forward_backward | 13.101 | 11.256 | 13.404 | 254.3 | 254.3 |

## Top CUDA Rows

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 5.301 | 0.5301 | 5.300 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.490 | 0.6980 | 3.490 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.810 | 0.3620 | 1.810 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.564 | 0.0261 | 1.564 |
| 5 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.065 | 0.0532 | 1.065 |
| 6 | aten::mean | 10 | 0.554 | 0.0554 | 0.544 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.534 | 0.0534 | 0.534 |
| 8 | cudaMemsetAsync | 90 | 0.394 | 0.0044 | 0.394 |
| 9 | aten::linalg_inv | 5 | 0.322 | 0.0644 | 0.000 |
| 10 | aten::linalg_inv_ex | 5 | 0.267 | 0.0534 | 0.000 |
| 11 | aten::inverse | 5 | 0.254 | 0.0508 | 0.000 |
| 12 | Memset (Device) | 90 | 0.244 | 0.0027 | 0.244 |
| 13 | aten::linalg_solve_ex | 5 | 0.236 | 0.0472 | 0.000 |
| 14 | aten::_linalg_solve_ex | 5 | 0.236 | 0.0472 | 0.000 |
| 15 | cudaLaunchKernel | 305 | 0.223 | 0.0007 | 0.223 |
| 16 | aten::cumsum | 10 | 0.211 | 0.0211 | 0.092 |
| 17 | aten::copy_ | 30 | 0.167 | 0.0056 | 0.140 |
| 18 | aten::linalg_lu_solve | 5 | 0.143 | 0.0286 | 0.120 |
| 19 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 10 | 0.129 | 0.0129 | 0.129 |
| 20 | aten::gt | 10 | 0.125 | 0.0125 | 0.040 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 31.941 | 3.1941 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 31.941 | 3.1941 | 31.772 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 22.405 | 4.4810 | 22.405 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 9.367 | 1.8734 | 9.367 |
| 5 | _RasterizeToPixels | 10 | 5.281 | 0.5281 | 5.281 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.484 | 0.6968 | 3.484 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.797 | 0.3594 | 1.797 |
| 8 | aten::copy_ | 95 | 1.586 | 0.0167 | 1.586 |
| 9 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.564 | 0.0261 | 1.564 |
| 10 | autograd::engine::evaluate_function: SelectBackward0 | 10 | 1.481 | 0.1481 | 0.000 |
| 11 | SelectBackward0 | 10 | 1.481 | 0.1481 | 0.000 |
| 12 | aten::select_backward | 10 | 1.481 | 0.1481 | 0.000 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.127 | 0.0564 | 1.127 |
| 14 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.058 | 0.0529 | 1.058 |
| 15 | aten::fill_ | 200 | 1.038 | 0.0052 | 1.038 |
| 16 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 0.994 | 0.0054 | 0.994 |
| 17 | aten::zero_ | 180 | 0.980 | 0.0054 | 0.000 |
| 18 | aten::zeros | 95 | 0.744 | 0.0078 | 0.000 |
| 19 | aten::div | 35 | 0.586 | 0.0167 | 0.586 |
| 20 | aten::mean | 10 | 0.541 | 0.0541 | 0.541 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 7.259 | 1.4518 | 7.259 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 7.259 | 1.4518 | 7.259 |
| 3 | aten::mean | 10 | 1.185 | 0.1185 | 1.185 |
| 4 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 1.175 | 0.1175 | 1.175 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.783 | 0.0261 | 0.783 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.537 | 0.0537 | 0.537 |
| 7 | aten::inverse | 5 | 0.300 | 0.0600 | 0.000 |
| 8 | aten::linalg_inv | 5 | 0.300 | 0.0600 | 0.000 |
| 9 | aten::linalg_inv_ex | 5 | 0.258 | 0.0516 | 0.000 |
| 10 | aten::linalg_solve_ex | 5 | 0.229 | 0.0458 | 0.000 |
| 11 | aten::_linalg_solve_ex | 5 | 0.229 | 0.0458 | 0.000 |
| 12 | aten::cat | 25 | 0.211 | 0.0084 | 0.211 |
| 13 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 0.170 | 0.0113 | 0.170 |
| 14 | aten::linalg_lu_solve | 5 | 0.142 | 0.0284 | 0.120 |
| 15 | Memset (Device) | 50 | 0.133 | 0.0027 | 0.133 |
| 16 | aten::copy_ | 25 | 0.096 | 0.0038 | 0.096 |
| 17 | aten::fill_ | 30 | 0.089 | 0.0030 | 0.089 |
| 18 | aten::linalg_lu_factor_ex | 5 | 0.087 | 0.0174 | 0.072 |
| 19 | aten::cumsum | 5 | 0.078 | 0.0156 | 0.047 |
| 20 | aten::zero_ | 25 | 0.074 | 0.0030 | 0.000 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 70.118 | 14.0236 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 70.118 | 14.0236 | 70.031 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 70.031 | 14.0062 | 70.031 |
| 4 | aten::div | 35 | 11.285 | 0.3224 | 11.285 |
| 5 | autograd::engine::evaluate_function: MeanBackward0 | 10 | 11.169 | 1.1169 | 0.000 |
| 6 | MeanBackward0 | 10 | 11.169 | 1.1169 | 0.000 |
| 7 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 10 | 11.169 | 1.1169 | 11.169 |
| 8 | autograd::engine::evaluate_function: SliceBackward0 | 25 | 8.693 | 0.3477 | 0.000 |
| 9 | SliceBackward0 | 25 | 7.358 | 0.2943 | 0.000 |
| 10 | aten::slice_backward | 25 | 7.358 | 0.2943 | 0.000 |
| 11 | _RasterizeToPixels | 5 | 7.168 | 1.4336 | 7.168 |
| 12 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 7.168 | 1.4336 | 7.168 |
| 13 | aten::fill_ | 140 | 5.413 | 0.0387 | 5.413 |
| 14 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 5.383 | 0.0414 | 5.383 |
| 15 | aten::zero_ | 120 | 5.355 | 0.0446 | 0.000 |
| 16 | aten::zeros | 70 | 5.209 | 0.0744 | 0.000 |
| 17 | aten::copy_ | 70 | 3.872 | 0.0553 | 3.872 |
| 18 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 2.789 | 0.0930 | 2.789 |
| 19 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3>) | 20 | 1.385 | 0.0693 | 1.385 |
| 20 | autograd::engine::evaluate_function: SelectBackward0 | 5 | 1.375 | 0.2750 | 0.000 |