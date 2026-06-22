# Render Backend Profiling

- model_path: `output/lerf/figurines_gsplat_r1_hybrid`
- iteration: `30000`
- view_index: `0`
- image: `986x728`
- gaussians: `15095`
- timed iterations: `30` after `5` warmup
- profiler iterations: `5` after `5` warmup
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| gsplat_two_pass | forward | 4.159 | 2.975 | 12.018 | 90.2 | 90.2 |
| gsplat_two_pass | forward_backward | 13.479 | 9.037 | 22.003 | 180.4 | 180.4 |
| gsplat_padded_single | forward | 5.481 | 2.765 | 9.088 | 80.7 | 80.7 |
| gsplat_padded_single | forward_backward | 12.587 | 7.394 | 22.275 | 189.9 | 189.9 |
| gsplat_single | forward | 5.146 | 2.520 | 8.979 | 86.1 | 86.1 |
| gsplat_single | forward_backward | 12.251 | 7.180 | 21.865 | 179.3 | 179.3 |

## Top CUDA Rows

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 5.287 | 0.5287 | 5.286 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.508 | 0.7016 | 3.508 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.778 | 0.3556 | 1.778 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.564 | 0.0261 | 1.564 |
| 5 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.074 | 0.0537 | 1.074 |
| 6 | aten::mean | 10 | 0.557 | 0.0557 | 0.547 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.537 | 0.0537 | 0.537 |
| 8 | cudaMemsetAsync | 90 | 0.385 | 0.0043 | 0.385 |
| 9 | aten::inverse | 5 | 0.326 | 0.0652 | 0.000 |
| 10 | aten::linalg_inv | 5 | 0.326 | 0.0652 | 0.000 |
| 11 | aten::linalg_inv_ex | 5 | 0.268 | 0.0536 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.238 | 0.0476 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.238 | 0.0476 | 0.000 |
| 14 | Memset (Device) | 90 | 0.234 | 0.0026 | 0.234 |
| 15 | cudaLaunchKernel | 305 | 0.222 | 0.0007 | 0.222 |
| 16 | aten::cumsum | 10 | 0.215 | 0.0215 | 0.093 |
| 17 | aten::copy_ | 30 | 0.171 | 0.0057 | 0.144 |
| 18 | aten::linalg_lu_solve | 5 | 0.144 | 0.0288 | 0.121 |
| 19 | aten::_to_copy | 20 | 0.127 | 0.0063 | 0.000 |
| 20 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 10 | 0.127 | 0.0127 | 0.127 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 23.444 | 2.3444 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 23.444 | 2.3444 | 23.271 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 15.119 | 3.0238 | 15.119 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 8.152 | 1.6304 | 8.152 |
| 5 | _RasterizeToPixels | 10 | 5.310 | 0.5310 | 5.310 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.501 | 0.7002 | 3.501 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.809 | 0.3618 | 1.809 |
| 8 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.588 | 0.0265 | 1.588 |
| 9 | aten::copy_ | 95 | 1.582 | 0.0167 | 1.582 |
| 10 | autograd::engine::evaluate_function: SelectBackward0 | 10 | 1.482 | 0.1482 | 0.000 |
| 11 | SelectBackward0 | 10 | 1.482 | 0.1482 | 0.000 |
| 12 | aten::select_backward | 10 | 1.482 | 0.1482 | 0.000 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.125 | 0.0563 | 1.125 |
| 14 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 1.077 | 0.0539 | 1.077 |
| 15 | aten::fill_ | 200 | 1.053 | 0.0053 | 1.053 |
| 16 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 1.010 | 0.0055 | 1.010 |
| 17 | aten::zero_ | 180 | 0.999 | 0.0056 | 0.000 |
| 18 | aten::zeros | 95 | 0.777 | 0.0082 | 0.000 |
| 19 | aten::div | 35 | 0.582 | 0.0166 | 0.582 |
| 20 | aten::mean | 10 | 0.546 | 0.0546 | 0.546 |

### gsplat_padded_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 4.239 | 0.8478 | 4.239 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.239 | 0.8478 | 4.239 |
| 3 | aten::mean | 10 | 0.943 | 0.0943 | 0.943 |
| 4 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.933 | 0.0933 | 0.933 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.784 | 0.0261 | 0.784 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.535 | 0.0535 | 0.535 |
| 7 | aten::inverse | 5 | 0.299 | 0.0598 | 0.000 |
| 8 | aten::linalg_inv | 5 | 0.299 | 0.0598 | 0.000 |
| 9 | aten::linalg_inv_ex | 5 | 0.259 | 0.0518 | 0.000 |
| 10 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 11 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 12 | aten::cat | 15 | 0.145 | 0.0097 | 0.145 |
| 13 | aten::linalg_lu_solve | 5 | 0.142 | 0.0284 | 0.120 |
| 14 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 10 | 0.125 | 0.0125 | 0.125 |
| 15 | Memset (Device) | 50 | 0.124 | 0.0025 | 0.124 |
| 16 | aten::linalg_lu_factor_ex | 5 | 0.089 | 0.0178 | 0.074 |
| 17 | aten::copy_ | 25 | 0.087 | 0.0035 | 0.087 |
| 18 | aten::cumsum | 5 | 0.079 | 0.0158 | 0.049 |
| 19 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 5 | 0.065 | 0.0130 | 0.065 |
| 20 | aten::to | 25 | 0.057 | 0.0023 | 0.000 |

### gsplat_padded_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 17.152 | 3.4304 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 17.152 | 3.4304 | 17.067 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 17.067 | 3.4134 | 17.067 |
| 4 | _RasterizeToPixels | 5 | 4.244 | 0.8488 | 4.244 |
| 5 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.244 | 0.8488 | 4.244 |
| 6 | autograd::engine::evaluate_function: SliceBackward0 | 20 | 3.803 | 0.1902 | 0.000 |
| 7 | aten::copy_ | 65 | 2.713 | 0.0417 | 2.713 |
| 8 | SliceBackward0 | 20 | 2.458 | 0.1229 | 0.000 |
| 9 | aten::slice_backward | 20 | 2.458 | 0.1229 | 0.000 |
| 10 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 25 | 1.643 | 0.0657 | 1.643 |
| 11 | aten::fill_ | 125 | 1.608 | 0.0129 | 1.608 |
| 12 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 115 | 1.577 | 0.0137 | 1.577 |
| 13 | aten::zero_ | 105 | 1.555 | 0.0148 | 0.000 |
| 14 | aten::zeros | 55 | 1.423 | 0.0259 | 0.000 |
| 15 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3>) | 20 | 1.391 | 0.0696 | 1.391 |
| 16 | autograd::engine::evaluate_function: SelectBackward0 | 5 | 1.382 | 0.2764 | 0.000 |
| 17 | SelectBackward0 | 5 | 1.382 | 0.2764 | 0.000 |
| 18 | aten::select_backward | 5 | 1.382 | 0.2764 | 0.000 |
| 19 | aten::add_ | 10 | 1.360 | 0.1360 | 1.360 |
| 20 | Memcpy DtoD (Device -> Device) | 25 | 1.013 | 0.0405 | 1.013 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 5.274 | 0.5274 | 5.274 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.501 | 0.7002 | 3.501 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.773 | 0.3546 | 1.773 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.790 | 0.0263 | 0.790 |
| 5 | aten::mean | 10 | 0.545 | 0.0545 | 0.545 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.536 | 0.0536 | 0.536 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.534 | 0.0534 | 0.534 |
| 8 | aten::inverse | 5 | 0.300 | 0.0600 | 0.000 |
| 9 | aten::linalg_inv | 5 | 0.300 | 0.0600 | 0.000 |
| 10 | aten::linalg_inv_ex | 5 | 0.260 | 0.0520 | 0.000 |
| 11 | aten::linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 12 | aten::_linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 13 | aten::linalg_lu_solve | 5 | 0.144 | 0.0288 | 0.120 |
| 14 | Memset (Device) | 50 | 0.124 | 0.0025 | 0.124 |
| 15 | aten::linalg_lu_factor_ex | 5 | 0.088 | 0.0176 | 0.073 |
| 16 | aten::copy_ | 25 | 0.086 | 0.0034 | 0.086 |
| 17 | aten::cat | 5 | 0.080 | 0.0160 | 0.080 |
| 18 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 0.080 | 0.0160 | 0.080 |
| 19 | aten::cumsum | 5 | 0.078 | 0.0156 | 0.048 |
| 20 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int>(int*, long const*, int, int, int) | 5 | 0.065 | 0.0130 | 0.065 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 23.547 | 2.3547 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 23.497 | 2.3497 | 23.333 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 16.087 | 3.2174 | 16.087 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 7.246 | 1.4492 | 7.246 |
| 5 | _RasterizeToPixels | 10 | 5.291 | 0.5291 | 5.291 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.491 | 0.6982 | 3.491 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 1.800 | 0.3600 | 1.800 |
| 8 | autograd::engine::evaluate_function: SelectBackward0 | 10 | 1.491 | 0.1491 | 0.000 |
| 9 | SelectBackward0 | 10 | 1.491 | 0.1491 | 0.000 |
| 10 | aten::select_backward | 10 | 1.491 | 0.1491 | 0.000 |
| 11 | aten::copy_ | 55 | 1.247 | 0.0227 | 1.247 |
| 12 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.124 | 0.0562 | 1.124 |
| 13 | aten::fill_ | 140 | 0.823 | 0.0059 | 0.823 |
| 14 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 0.793 | 0.0061 | 0.793 |
| 15 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.789 | 0.0263 | 0.789 |
| 16 | aten::zero_ | 120 | 0.768 | 0.0064 | 0.000 |
| 17 | aten::zeros | 50 | 0.583 | 0.0117 | 0.000 |
| 18 | aten::div | 35 | 0.581 | 0.0166 | 0.581 |
| 19 | aten::mean | 10 | 0.546 | 0.0546 | 0.546 |
| 20 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.536 | 0.0536 | 0.536 |