# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- image: `493x364`
- gaussians: `280344`
- timed iterations: `10` after `3` warmup
- profiler iterations: `5` after `3` warmup

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| diff | forward | 3.748 | 2.572 | 4.810 | 226.8 | 226.8 |
| diff | forward_backward | 15.762 | 14.417 | 16.206 | 342.0 | 342.0 |
| gsplat_two_pass | forward | 4.446 | 3.065 | 5.913 | 214.2 | 214.2 |
| gsplat_two_pass | forward_backward | 11.480 | 10.889 | 11.781 | 276.8 | 276.8 |
| gsplat_single | forward | 4.524 | 3.037 | 9.537 | 260.6 | 260.6 |
| gsplat_single | forward_backward | 12.023 | 9.480 | 17.963 | 297.0 | 297.0 |

## Top CUDA Rows

### diff_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeGaussians | 5 | 13.340 | 2.6680 | 9.132 |
| 2 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 6.316 | 1.2632 | 6.316 |
| 3 | cudaPeekAtLastError | 60 | 2.148 | 0.0358 | 2.148 |
| 4 | cudaMemsetAsync | 50 | 1.835 | 0.0367 | 1.835 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.398 | 0.0466 | 1.398 |
| 6 | aten::cat | 5 | 1.366 | 0.2732 | 1.366 |
| 7 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.366 | 0.2732 | 1.366 |
| 8 | void preprocessCUDA<3, 16>(int, int, int, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float, glm::vec<4, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool*, float const*, float const*, float const*, float const*, glm::vec<3, float, (glm::qualifier)0> const*, int, int, float, float, float, float, int*, float2*, float*, float*, float*, float4*, dim3, unsigned int*, bool) | 5 | 0.524 | 0.1048 | 0.524 |
| 9 | duplicateWithKeys(int, float2 const*, float const*, unsigned int const*, unsigned long*, unsigned int*, int*, dim3) | 5 | 0.494 | 0.0988 | 0.494 |
| 10 | aten::mean | 10 | 0.211 | 0.0211 | 0.211 |
| 11 | aten::fill_ | 20 | 0.201 | 0.0100 | 0.177 |
| 12 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.201 | 0.0201 | 0.201 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 15 | 0.157 | 0.0105 | 0.157 |
| 14 | Memset (Device) | 55 | 0.138 | 0.0025 | 0.138 |
| 15 | aten::full | 15 | 0.126 | 0.0084 | 0.000 |
| 16 | aten::linalg_vector_norm | 5 | 0.114 | 0.0228 | 0.114 |
| 17 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4>) | 5 | 0.114 | 0.0228 | 0.114 |
| 18 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, int>(int*, unsigned long const*, int, int, int) | 5 | 0.108 | 0.0216 | 0.108 |
| 19 | aten::div | 5 | 0.104 | 0.0208 | 0.097 |
| 20 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 5 | 0.097 | 0.0194 | 0.097 |

### diff_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeGaussiansBackward | 5 | 97.145 | 19.4290 | 0.000 |
| 2 | _RasterizeGaussiansBackward | 5 | 97.145 | 19.4290 | 96.183 |
| 3 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float const*, float2 const*, float4 const*, float const*, float const*, float const*, unsigned int const*, float const*, float const*, float3*, float4*, float*, float*, float*) | 5 | 94.559 | 18.9118 | 94.559 |
| 4 | aten::copy_ | 35 | 10.725 | 0.3064 | 10.725 |
| 5 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 10.630 | 0.3543 | 10.630 |
| 6 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 40 | 10.562 | 0.2641 | 0.000 |
| 7 | torch::autograd::AccumulateGrad | 40 | 10.562 | 0.2641 | 0.000 |
| 8 | _RasterizeGaussians | 5 | 9.373 | 1.8746 | 9.195 |
| 9 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 6.302 | 1.2604 | 6.302 |
| 10 | aten::div | 35 | 3.252 | 0.0929 | 3.252 |
| 11 | autograd::engine::evaluate_function: DivBackward0 | 5 | 3.126 | 0.6252 | 0.000 |
| 12 | DivBackward0 | 5 | 3.126 | 0.6252 | 0.000 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 25 | 3.107 | 0.1243 | 3.107 |
| 14 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.435 | 0.0478 | 1.435 |
| 15 | void preprocessCUDA<3>(int, int, int, float3 const*, int const*, float const*, bool const*, glm::vec<3, float, (glm::qualifier)0> const*, glm::vec<4, float, (glm::qualifier)0> const*, float, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float3 const*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*, float*, glm::vec<3, float, (glm::qualifier)0>*, glm::vec<4, float, (glm::qualifier)0>*) | 5 | 1.430 | 0.2860 | 1.430 |
| 16 | aten::cat | 5 | 1.376 | 0.2752 | 1.376 |
| 17 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.376 | 0.2752 | 1.376 |
| 18 | aten::fill_ | 85 | 1.138 | 0.0134 | 1.138 |
| 19 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 75 | 1.096 | 0.0146 | 1.096 |
| 20 | aten::zero_ | 60 | 0.962 | 0.0160 | 0.000 |

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.239 | 0.6239 | 6.239 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.432 | 0.6864 | 3.432 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.807 | 0.5614 | 2.807 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.619 | 0.0270 | 1.619 |
| 5 | aten::cat | 5 | 1.369 | 0.2738 | 1.369 |
| 6 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.369 | 0.2738 | 1.369 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 0.663 | 0.0331 | 0.663 |
| 8 | _FullyFusedProjection | 10 | 0.424 | 0.0424 | 0.424 |
| 9 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 10 | 0.424 | 0.0424 | 0.424 |
| 10 | aten::inverse | 5 | 0.300 | 0.0600 | 0.000 |
| 11 | aten::linalg_inv | 5 | 0.300 | 0.0600 | 0.000 |
| 12 | aten::linalg_inv_ex | 5 | 0.262 | 0.0524 | 0.000 |
| 13 | Memset (Device) | 90 | 0.250 | 0.0028 | 0.250 |
| 14 | aten::mean | 10 | 0.239 | 0.0239 | 0.239 |
| 15 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 16 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 17 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.223 | 0.0223 | 0.223 |
| 18 | _SphericalHarmonics | 5 | 0.219 | 0.0438 | 0.219 |
| 19 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.219 | 0.0438 | 0.219 |
| 20 | aten::cumsum | 10 | 0.215 | 0.0215 | 0.134 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 34.116 | 3.4116 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 34.116 | 3.4116 | 33.736 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 23.325 | 4.6650 | 23.325 |
| 4 | autograd::engine::evaluate_function: SliceBackward0 | 45 | 11.477 | 0.2550 | 0.000 |
| 5 | SliceBackward0 | 45 | 11.477 | 0.2550 | 0.000 |
| 6 | aten::slice_backward | 45 | 11.477 | 0.2550 | 0.000 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 10.411 | 2.0822 | 10.411 |
| 8 | aten::copy_ | 95 | 8.283 | 0.0872 | 8.283 |
| 9 | Memcpy DtoD (Device -> Device) | 55 | 6.776 | 0.1232 | 6.776 |
| 10 | _RasterizeToPixels | 10 | 6.260 | 0.6260 | 6.260 |
| 11 | aten::fill_ | 200 | 6.160 | 0.0308 | 6.160 |
| 12 | aten::zero_ | 180 | 6.100 | 0.0339 | 0.000 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 6.093 | 0.0329 | 6.093 |
| 14 | aten::zeros | 95 | 5.033 | 0.0530 | 0.000 |
| 15 | autograd::engine::evaluate_function: _FullyFusedProjectionBackward | 10 | 4.357 | 0.4357 | 0.000 |
| 16 | _FullyFusedProjectionBackward | 10 | 3.946 | 0.3946 | 3.621 |
| 17 | void gsplat::projection_ewa_3dgs_fused_bwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, gsplat::CameraModelType, int const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*) | 10 | 3.621 | 0.3621 | 3.621 |
| 18 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 3.604 | 0.7208 | 0.000 |
| 19 | _SphericalHarmonicsBackward | 5 | 3.604 | 0.7208 | 3.140 |
| 20 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3>) | 35 | 3.518 | 0.1005 | 3.518 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 4.926 | 0.9852 | 4.926 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.926 | 0.9852 | 4.926 |
| 3 | aten::cat | 25 | 2.843 | 0.1137 | 2.843 |
| 4 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 2.803 | 0.1869 | 2.803 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.805 | 0.0268 | 0.805 |
| 6 | aten::mean | 10 | 0.386 | 0.0386 | 0.386 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.375 | 0.0375 | 0.375 |
| 8 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.335 | 0.0335 | 0.335 |
| 9 | aten::inverse | 5 | 0.299 | 0.0598 | 0.000 |
| 10 | aten::linalg_inv | 5 | 0.299 | 0.0598 | 0.000 |
| 11 | aten::linalg_inv_ex | 5 | 0.259 | 0.0518 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.230 | 0.0460 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.230 | 0.0460 | 0.000 |
| 14 | _SphericalHarmonics | 5 | 0.223 | 0.0446 | 0.223 |
| 15 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.223 | 0.0446 | 0.223 |
| 16 | _FullyFusedProjection | 5 | 0.219 | 0.0438 | 0.219 |
| 17 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.219 | 0.0438 | 0.219 |
| 18 | aten::fill_ | 30 | 0.205 | 0.0068 | 0.205 |
| 19 | aten::zero_ | 25 | 0.190 | 0.0076 | 0.000 |
| 20 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 25 | 0.190 | 0.0076 | 0.190 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 51.137 | 10.2274 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 51.137 | 10.2274 | 44.518 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 44.518 | 8.9036 | 44.518 |
| 4 | aten::fill_ | 140 | 8.127 | 0.0580 | 8.127 |
| 5 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 8.082 | 0.0622 | 8.082 |
| 6 | aten::zero_ | 120 | 8.067 | 0.0672 | 0.000 |
| 7 | aten::copy_ | 70 | 7.819 | 0.1117 | 7.819 |
| 8 | aten::zeros_like | 45 | 7.184 | 0.1596 | 0.000 |
| 9 | autograd::engine::evaluate_function: SliceBackward0 | 25 | 6.726 | 0.2690 | 0.000 |
| 10 | SliceBackward0 | 25 | 6.381 | 0.2552 | 0.000 |
| 11 | aten::slice_backward | 25 | 6.381 | 0.2552 | 0.000 |
| 12 | Memcpy DtoD (Device -> Device) | 25 | 5.474 | 0.2190 | 5.474 |
| 13 | _RasterizeToPixels | 5 | 4.900 | 0.9800 | 4.900 |
| 14 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.900 | 0.9800 | 4.900 |
| 15 | aten::div | 35 | 3.589 | 0.1025 | 3.589 |
| 16 | autograd::engine::evaluate_function: DivBackward0 | 5 | 3.462 | 0.6924 | 0.000 |
| 17 | DivBackward0 | 5 | 3.462 | 0.6924 | 0.000 |
| 18 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 25 | 3.443 | 0.1377 | 3.443 |
| 19 | aten::cat | 30 | 2.905 | 0.0968 | 2.905 |
| 20 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 2.784 | 0.1856 | 2.784 |