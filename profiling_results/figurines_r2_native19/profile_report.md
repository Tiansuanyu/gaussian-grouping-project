# Render Backend Profiling

- model_path: `output/lerf/figurines`
- iteration: `30000`
- view_index: `0`
- image: `493x364`
- gaussians: `280344`
- timed iterations: `10` after `3` warmup
- profiler iterations: `5` after `3` warmup
- chart: [`summary_chart.svg`](summary_chart.svg)

## Summary

| Backend | Mode | Mean ms | Min ms | Max ms | Peak MB | Profiler Peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| diff | forward | 4.472 | 3.487 | 4.860 | 226.8 | 226.8 |
| diff | forward_backward | 18.659 | 15.134 | 28.470 | 342.0 | 342.0 |
| gsplat_two_pass | forward | 10.518 | 3.712 | 12.542 | 214.2 | 214.2 |
| gsplat_two_pass | forward_backward | 10.504 | 9.639 | 11.755 | 276.8 | 276.8 |
| gsplat_padded_single | forward | 2.725 | 2.434 | 3.571 | 217.5 | 217.5 |
| gsplat_padded_single | forward_backward | 8.227 | 6.810 | 10.146 | 263.5 | 263.5 |
| gsplat_single | forward | 7.036 | 3.794 | 9.454 | 198.9 | 198.9 |
| gsplat_single | forward_backward | 9.697 | 8.448 | 11.171 | 264.9 | 264.9 |

## Top CUDA Rows

### diff_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeGaussians | 5 | 13.306 | 2.6612 | 9.112 |
| 2 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 6.272 | 1.2544 | 6.272 |
| 3 | cudaPeekAtLastError | 60 | 2.132 | 0.0355 | 2.132 |
| 4 | cudaMemsetAsync | 50 | 2.124 | 0.0425 | 1.839 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.402 | 0.0467 | 1.402 |
| 6 | aten::cat | 5 | 1.382 | 0.2764 | 1.382 |
| 7 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.382 | 0.2764 | 1.382 |
| 8 | void preprocessCUDA<3, 16>(int, int, int, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float, glm::vec<4, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool*, float const*, float const*, float const*, float const*, glm::vec<3, float, (glm::qualifier)0> const*, int, int, float, float, float, float, int*, float2*, float*, float*, float*, float4*, dim3, unsigned int*, bool) | 5 | 0.525 | 0.1050 | 0.525 |
| 9 | duplicateWithKeys(int, float2 const*, float const*, unsigned int const*, unsigned long*, unsigned int*, int*, dim3) | 5 | 0.498 | 0.0996 | 0.498 |
| 10 | aten::mean | 10 | 0.215 | 0.0215 | 0.215 |
| 11 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.204 | 0.0204 | 0.204 |
| 12 | aten::fill_ | 20 | 0.198 | 0.0099 | 0.173 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 15 | 0.153 | 0.0102 | 0.153 |
| 14 | Memset (Device) | 55 | 0.152 | 0.0028 | 0.152 |
| 15 | aten::full | 15 | 0.145 | 0.0097 | 0.000 |
| 16 | aten::linalg_vector_norm | 5 | 0.117 | 0.0234 | 0.117 |
| 17 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4>) | 5 | 0.117 | 0.0234 | 0.117 |
| 18 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, int>(int*, unsigned long const*, int, int, int) | 5 | 0.109 | 0.0218 | 0.109 |
| 19 | aten::div | 5 | 0.107 | 0.0214 | 0.099 |
| 20 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 5 | 0.099 | 0.0198 | 0.099 |

### diff_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeGaussiansBackward | 5 | 93.482 | 18.6964 | 0.000 |
| 2 | _RasterizeGaussiansBackward | 5 | 93.482 | 18.6964 | 92.510 |
| 3 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float const*, float2 const*, float4 const*, float const*, float const*, float const*, unsigned int const*, float const*, float const*, float3*, float4*, float*, float*, float*) | 5 | 90.868 | 18.1736 | 90.868 |
| 4 | _RasterizeGaussians | 5 | 9.350 | 1.8700 | 9.167 |
| 5 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 6.266 | 1.2532 | 6.266 |
| 6 | aten::div | 35 | 4.715 | 0.1347 | 4.715 |
| 7 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 25 | 4.570 | 0.1828 | 4.570 |
| 8 | autograd::engine::evaluate_function: LinalgVectorNormBackward0 | 5 | 4.500 | 0.9000 | 0.000 |
| 9 | LinalgVectorNormBackward0 | 5 | 4.370 | 0.8740 | 0.000 |
| 10 | void preprocessCUDA<3>(int, int, int, float3 const*, int const*, float const*, bool const*, glm::vec<3, float, (glm::qualifier)0> const*, glm::vec<4, float, (glm::qualifier)0> const*, float, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float3 const*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*, float*, glm::vec<3, float, (glm::qualifier)0>*, glm::vec<4, float, (glm::qualifier)0>*) | 5 | 1.448 | 0.2896 | 1.448 |
| 11 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.430 | 0.0477 | 1.430 |
| 12 | aten::cat | 5 | 1.382 | 0.2764 | 1.382 |
| 13 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.382 | 0.2764 | 1.382 |
| 14 | aten::copy_ | 35 | 1.262 | 0.0361 | 1.262 |
| 15 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 1.168 | 0.0389 | 1.168 |
| 16 | aten::fill_ | 85 | 1.150 | 0.0135 | 1.150 |
| 17 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 75 | 1.100 | 0.0147 | 1.100 |
| 18 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 40 | 1.095 | 0.0274 | 0.000 |
| 19 | torch::autograd::AccumulateGrad | 40 | 1.095 | 0.0274 | 0.000 |
| 20 | aten::zero_ | 60 | 0.973 | 0.0162 | 0.000 |

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.216 | 0.6216 | 6.216 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.497 | 0.6994 | 3.497 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.719 | 0.5438 | 2.719 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.618 | 0.0270 | 1.618 |
| 5 | aten::cat | 5 | 1.377 | 0.2754 | 1.377 |
| 6 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.377 | 0.2754 | 1.377 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 0.673 | 0.0336 | 0.673 |
| 8 | _FullyFusedProjection | 10 | 0.430 | 0.0430 | 0.430 |
| 9 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 10 | 0.430 | 0.0430 | 0.430 |
| 10 | aten::inverse | 5 | 0.306 | 0.0612 | 0.000 |
| 11 | aten::linalg_inv | 5 | 0.306 | 0.0612 | 0.000 |
| 12 | aten::linalg_inv_ex | 5 | 0.262 | 0.0524 | 0.000 |
| 13 | aten::linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 14 | aten::_linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 15 | Memset (Device) | 90 | 0.232 | 0.0026 | 0.232 |
| 16 | aten::cumsum | 10 | 0.227 | 0.0227 | 0.138 |
| 17 | aten::mean | 10 | 0.225 | 0.0225 | 0.225 |
| 18 | _SphericalHarmonics | 5 | 0.220 | 0.0440 | 0.220 |
| 19 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.220 | 0.0440 | 0.220 |
| 20 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.214 | 0.0214 | 0.214 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 30.482 | 3.0482 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 30.482 | 3.0482 | 30.096 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 19.628 | 3.9256 | 19.628 |
| 4 | autograd::engine::evaluate_function: SliceBackward0 | 45 | 11.156 | 0.2479 | 0.000 |
| 5 | SliceBackward0 | 45 | 11.156 | 0.2479 | 0.000 |
| 6 | aten::slice_backward | 45 | 11.156 | 0.2479 | 0.000 |
| 7 | aten::copy_ | 95 | 11.022 | 0.1160 | 11.022 |
| 8 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 10.468 | 2.0936 | 10.468 |
| 9 | Memcpy DtoD (Device -> Device) | 55 | 9.511 | 0.1729 | 9.511 |
| 10 | _RasterizeToPixels | 10 | 6.199 | 0.6199 | 6.199 |
| 11 | autograd::engine::evaluate_function: _FullyFusedProjectionBackward | 10 | 4.285 | 0.4285 | 0.000 |
| 12 | _FullyFusedProjectionBackward | 10 | 3.872 | 0.3872 | 3.548 |
| 13 | void gsplat::projection_ewa_3dgs_fused_bwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, gsplat::CameraModelType, int const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*) | 10 | 3.548 | 0.3548 | 3.548 |
| 14 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.441 | 0.6882 | 3.441 |
| 15 | aten::fill_ | 200 | 3.098 | 0.0155 | 3.098 |
| 16 | aten::zero_ | 180 | 3.039 | 0.0169 | 0.000 |
| 17 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 3.034 | 0.0164 | 3.034 |
| 18 | autograd::engine::evaluate_function: LinalgVectorNormBackward0 | 5 | 2.996 | 0.5992 | 0.000 |
| 19 | LinalgVectorNormBackward0 | 5 | 2.871 | 0.5742 | 0.000 |
| 20 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.758 | 0.5516 | 2.758 |

### gsplat_padded_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 4.010 | 0.8020 | 4.010 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.010 | 0.8020 | 4.010 |
| 3 | aten::cat | 15 | 1.973 | 0.1315 | 1.973 |
| 4 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 10 | 1.953 | 0.1953 | 1.953 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.802 | 0.0267 | 0.802 |
| 6 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.331 | 0.0331 | 0.331 |
| 7 | aten::mean | 10 | 0.327 | 0.0327 | 0.327 |
| 8 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.317 | 0.0317 | 0.317 |
| 9 | aten::inverse | 5 | 0.307 | 0.0614 | 0.000 |
| 10 | aten::linalg_inv | 5 | 0.307 | 0.0614 | 0.000 |
| 11 | aten::linalg_inv_ex | 5 | 0.266 | 0.0532 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.236 | 0.0472 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.236 | 0.0472 | 0.000 |
| 14 | _SphericalHarmonics | 5 | 0.226 | 0.0452 | 0.226 |
| 15 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.226 | 0.0452 | 0.226 |
| 16 | _FullyFusedProjection | 5 | 0.214 | 0.0428 | 0.214 |
| 17 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.214 | 0.0428 | 0.214 |
| 18 | aten::linalg_lu_solve | 5 | 0.148 | 0.0296 | 0.123 |
| 19 | Memset (Device) | 50 | 0.125 | 0.0025 | 0.125 |
| 20 | aten::linalg_vector_norm | 5 | 0.122 | 0.0244 | 0.122 |

### gsplat_padded_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 19.507 | 3.9014 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 19.507 | 3.9014 | 19.232 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 19.232 | 3.8464 | 19.232 |
| 4 | _RasterizeToPixels | 5 | 3.923 | 0.7846 | 3.923 |
| 5 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<19u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.923 | 0.7846 | 3.923 |
| 6 | aten::copy_ | 65 | 2.398 | 0.0369 | 2.398 |
| 7 | aten::div | 35 | 2.179 | 0.0623 | 2.179 |
| 8 | aten::cat | 20 | 2.054 | 0.1027 | 2.054 |
| 9 | autograd::engine::evaluate_function: DivBackward0 | 5 | 2.049 | 0.4098 | 0.000 |
| 10 | DivBackward0 | 5 | 2.049 | 0.4098 | 0.000 |
| 11 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 25 | 2.032 | 0.0813 | 2.032 |
| 12 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 10 | 1.952 | 0.1952 | 1.952 |
| 13 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 25 | 1.846 | 0.0738 | 1.846 |
| 14 | aten::fill_ | 125 | 1.459 | 0.0117 | 1.459 |
| 15 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 1.436 | 0.0410 | 0.000 |
| 16 | torch::autograd::AccumulateGrad | 35 | 1.436 | 0.0410 | 0.000 |
| 17 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 115 | 1.418 | 0.0123 | 1.418 |
| 18 | aten::zero_ | 105 | 1.403 | 0.0134 | 0.000 |
| 19 | autograd::engine::evaluate_function: SliceBackward0 | 20 | 1.235 | 0.0617 | 0.000 |
| 20 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 1.059 | 0.2118 | 0.000 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.182 | 0.6182 | 6.182 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.479 | 0.6958 | 3.479 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.703 | 0.5406 | 2.703 |
| 4 | aten::cat | 5 | 1.363 | 0.2726 | 1.363 |
| 5 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.363 | 0.2726 | 1.363 |
| 6 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.813 | 0.0271 | 0.813 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.338 | 0.0338 | 0.338 |
| 8 | aten::inverse | 5 | 0.299 | 0.0598 | 0.000 |
| 9 | aten::linalg_inv | 5 | 0.299 | 0.0598 | 0.000 |
| 10 | aten::linalg_inv_ex | 5 | 0.260 | 0.0520 | 0.000 |
| 11 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 12 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 13 | _SphericalHarmonics | 5 | 0.224 | 0.0448 | 0.224 |
| 14 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.224 | 0.0448 | 0.224 |
| 15 | aten::mean | 10 | 0.221 | 0.0221 | 0.221 |
| 16 | _FullyFusedProjection | 5 | 0.214 | 0.0428 | 0.214 |
| 17 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.214 | 0.0428 | 0.214 |
| 18 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.211 | 0.0211 | 0.211 |
| 19 | aten::linalg_lu_solve | 5 | 0.143 | 0.0286 | 0.121 |
| 20 | aten::linalg_vector_norm | 5 | 0.120 | 0.0240 | 0.120 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 40.574 | 4.0574 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 40.369 | 4.0369 | 39.977 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 30.086 | 6.0172 | 30.086 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 9.891 | 1.9782 | 9.891 |
| 5 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 7.508 | 1.5016 | 0.000 |
| 6 | _SphericalHarmonicsBackward | 5 | 7.508 | 1.5016 | 6.283 |
| 7 | void gsplat::spherical_harmonics_bwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float const*, float*, float*) | 5 | 6.283 | 1.2566 | 6.283 |
| 8 | _RasterizeToPixels | 10 | 6.129 | 0.6129 | 6.129 |
| 9 | aten::copy_ | 55 | 4.704 | 0.0855 | 4.704 |
| 10 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 4.413 | 0.2207 | 4.413 |
| 11 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 4.122 | 0.1178 | 0.000 |
| 12 | torch::autograd::AccumulateGrad | 35 | 4.122 | 0.1178 | 0.000 |
| 13 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.418 | 0.6836 | 3.418 |
| 14 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.711 | 0.5422 | 2.711 |
| 15 | aten::fill_ | 140 | 2.108 | 0.0151 | 2.108 |
| 16 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 2.068 | 0.0159 | 2.068 |
| 17 | aten::zero_ | 120 | 2.040 | 0.0170 | 0.000 |
| 18 | aten::zeros_like | 65 | 1.698 | 0.0261 | 0.000 |
| 19 | aten::cat | 10 | 1.441 | 0.1441 | 1.441 |
| 20 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.362 | 0.2724 | 1.362 |