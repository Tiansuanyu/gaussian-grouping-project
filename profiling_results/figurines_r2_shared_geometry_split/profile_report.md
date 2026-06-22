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
| diff | forward | 2.633 | 2.597 | 2.656 | 226.8 | 226.8 |
| diff | forward_backward | 13.186 | 12.755 | 13.710 | 342.0 | 342.0 |
| gsplat_two_pass | forward | 3.017 | 2.978 | 3.080 | 214.2 | 214.2 |
| gsplat_two_pass | forward_backward | 9.397 | 9.293 | 9.543 | 276.8 | 276.8 |
| gsplat_padded_single | forward | 2.796 | 2.767 | 2.835 | 260.6 | 260.6 |
| gsplat_padded_single | forward_backward | 9.401 | 9.343 | 9.449 | 297.0 | 297.0 |
| gsplat_single | forward | 2.710 | 2.674 | 2.748 | 198.9 | 198.9 |
| gsplat_single | forward_backward | 7.877 | 7.781 | 7.988 | 264.9 | 264.9 |

## Top CUDA Rows

### diff_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeGaussians | 5 | 14.637 | 2.9274 | 10.023 |
| 2 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 7.003 | 1.4006 | 7.003 |
| 3 | cudaPeekAtLastError | 60 | 2.331 | 0.0389 | 2.331 |
| 4 | cudaMemsetAsync | 50 | 2.052 | 0.0410 | 2.048 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.497 | 0.0499 | 1.497 |
| 6 | aten::cat | 5 | 1.386 | 0.2772 | 1.386 |
| 7 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.386 | 0.2772 | 1.386 |
| 8 | void preprocessCUDA<3, 16>(int, int, int, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float, glm::vec<4, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool*, float const*, float const*, float const*, float const*, glm::vec<3, float, (glm::qualifier)0> const*, int, int, float, float, float, float, int*, float2*, float*, float*, float*, float4*, dim3, unsigned int*, bool) | 5 | 0.548 | 0.1096 | 0.548 |
| 9 | duplicateWithKeys(int, float2 const*, float const*, unsigned int const*, unsigned long*, unsigned int*, int*, dim3) | 5 | 0.541 | 0.1082 | 0.541 |
| 10 | aten::mean | 10 | 0.217 | 0.0217 | 0.217 |
| 11 | aten::fill_ | 20 | 0.208 | 0.0104 | 0.181 |
| 12 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.207 | 0.0207 | 0.207 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 15 | 0.159 | 0.0106 | 0.159 |
| 14 | Memset (Device) | 55 | 0.159 | 0.0029 | 0.159 |
| 15 | aten::full | 15 | 0.150 | 0.0100 | 0.000 |
| 16 | aten::linalg_vector_norm | 5 | 0.126 | 0.0252 | 0.126 |
| 17 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4>) | 5 | 0.126 | 0.0252 | 0.126 |
| 18 | void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, int>(int*, unsigned long const*, int, int, int) | 5 | 0.114 | 0.0228 | 0.114 |
| 19 | aten::div | 5 | 0.108 | 0.0216 | 0.100 |
| 20 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 5 | 0.100 | 0.0200 | 0.100 |

### diff_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeGaussiansBackward | 5 | 48.451 | 9.6902 | 0.000 |
| 2 | _RasterizeGaussiansBackward | 5 | 48.451 | 9.6902 | 47.497 |
| 3 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float const*, float2 const*, float4 const*, float const*, float const*, float const*, unsigned int const*, float const*, float const*, float3*, float4*, float*, float*, float*) | 5 | 45.841 | 9.1682 | 45.841 |
| 4 | _RasterizeGaussians | 5 | 9.521 | 1.9042 | 9.346 |
| 5 | void renderCUDA<3u, 16u>(uint2 const*, unsigned int const*, int, int, float2 const*, float const*, float const*, float4 const*, float*, unsigned int*, float const*, float*, float*) | 5 | 6.428 | 1.2856 | 6.428 |
| 6 | void preprocessCUDA<3>(int, int, int, float3 const*, int const*, float const*, bool const*, glm::vec<3, float, (glm::qualifier)0> const*, glm::vec<4, float, (glm::qualifier)0> const*, float, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float3 const*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*, float*, glm::vec<3, float, (glm::qualifier)0>*, glm::vec<4, float, (glm::qualifier)0>*) | 5 | 1.463 | 0.2926 | 1.463 |
| 7 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<unsigned long, unsigned int, int>::Policy800, false, unsigned long, unsigned int, int, int>(int*, int*, int*, int const*, unsigned long*, unsigned long const*, unsigned int*, unsigned int const*, int, int, int) | 30 | 1.440 | 0.0480 | 1.440 |
| 8 | aten::cat | 5 | 1.378 | 0.2756 | 1.378 |
| 9 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.378 | 0.2756 | 1.378 |
| 10 | aten::copy_ | 35 | 1.241 | 0.0355 | 1.241 |
| 11 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 1.147 | 0.0382 | 1.147 |
| 12 | aten::fill_ | 85 | 1.137 | 0.0134 | 1.137 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 75 | 1.097 | 0.0146 | 1.097 |
| 14 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 40 | 1.087 | 0.0272 | 0.000 |
| 15 | torch::autograd::AccumulateGrad | 40 | 1.087 | 0.0272 | 0.000 |
| 16 | aten::zero_ | 60 | 0.962 | 0.0160 | 0.000 |
| 17 | aten::zeros | 55 | 0.924 | 0.0168 | 0.000 |
| 18 | aten::div | 35 | 0.670 | 0.0191 | 0.670 |
| 19 | void preprocessCUDA<3, 16>(int, int, int, float const*, glm::vec<3, float, (glm::qualifier)0> const*, float, glm::vec<4, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool*, float const*, float const*, float const*, float const*, glm::vec<3, float, (glm::qualifier)0> const*, int, int, float, float, float, float, int*, float2*, float*, float*, float*, float4*, dim3, unsigned int*, bool) | 5 | 0.546 | 0.1092 | 0.546 |
| 20 | autograd::engine::evaluate_function: DivBackward0 | 5 | 0.536 | 0.1072 | 0.000 |

### gsplat_two_pass_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.328 | 0.6328 | 6.328 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.476 | 0.6952 | 3.476 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.852 | 0.5704 | 2.852 |
| 4 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.585 | 0.0264 | 1.585 |
| 5 | aten::cat | 5 | 1.379 | 0.2758 | 1.379 |
| 6 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.379 | 0.2758 | 1.379 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 20 | 0.653 | 0.0326 | 0.653 |
| 8 | _FullyFusedProjection | 10 | 0.403 | 0.0403 | 0.403 |
| 9 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 10 | 0.403 | 0.0403 | 0.403 |
| 10 | aten::inverse | 5 | 0.296 | 0.0592 | 0.000 |
| 11 | aten::linalg_inv | 5 | 0.296 | 0.0592 | 0.000 |
| 12 | aten::linalg_inv_ex | 5 | 0.261 | 0.0522 | 0.000 |
| 13 | aten::linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 14 | aten::_linalg_solve_ex | 5 | 0.231 | 0.0462 | 0.000 |
| 15 | Memset (Device) | 90 | 0.228 | 0.0025 | 0.228 |
| 16 | aten::mean | 10 | 0.223 | 0.0223 | 0.223 |
| 17 | _SphericalHarmonics | 5 | 0.218 | 0.0436 | 0.218 |
| 18 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.218 | 0.0436 | 0.218 |
| 19 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.213 | 0.0213 | 0.213 |
| 20 | aten::cumsum | 10 | 0.192 | 0.0192 | 0.120 |

### gsplat_two_pass_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 20.345 | 2.0345 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 20.345 | 2.0345 | 19.965 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 12.641 | 2.5282 | 12.641 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 7.324 | 1.4648 | 7.324 |
| 5 | _RasterizeToPixels | 10 | 6.302 | 0.6302 | 6.302 |
| 6 | autograd::engine::evaluate_function: SliceBackward0 | 45 | 5.418 | 0.1204 | 0.000 |
| 7 | SliceBackward0 | 45 | 5.418 | 0.1204 | 0.000 |
| 8 | aten::slice_backward | 45 | 5.418 | 0.1204 | 0.000 |
| 9 | aten::copy_ | 95 | 5.240 | 0.0552 | 5.240 |
| 10 | Memcpy DtoD (Device -> Device) | 55 | 3.764 | 0.0684 | 3.764 |
| 11 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.495 | 0.6990 | 3.495 |
| 12 | aten::fill_ | 200 | 3.085 | 0.0154 | 3.085 |
| 13 | aten::zero_ | 180 | 3.030 | 0.0168 | 0.000 |
| 14 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 185 | 3.022 | 0.0163 | 3.022 |
| 15 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.807 | 0.5614 | 2.807 |
| 16 | aten::zeros | 95 | 1.960 | 0.0206 | 0.000 |
| 17 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 60 | 1.599 | 0.0267 | 1.599 |
| 18 | autograd::engine::evaluate_function: _FullyFusedProjectionBackward | 10 | 1.525 | 0.1525 | 0.000 |
| 19 | aten::cat | 10 | 1.456 | 0.1456 | 1.456 |
| 20 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.378 | 0.0689 | 1.378 |

### gsplat_padded_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 5 | 4.906 | 0.9812 | 4.906 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.906 | 0.9812 | 4.906 |
| 3 | aten::cat | 25 | 2.832 | 0.1133 | 2.832 |
| 4 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 2.791 | 0.1861 | 2.791 |
| 5 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.803 | 0.0268 | 0.803 |
| 6 | aten::mean | 10 | 0.385 | 0.0385 | 0.385 |
| 7 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.375 | 0.0375 | 0.375 |
| 8 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.326 | 0.0326 | 0.326 |
| 9 | aten::inverse | 5 | 0.301 | 0.0602 | 0.000 |
| 10 | aten::linalg_inv | 5 | 0.301 | 0.0602 | 0.000 |
| 11 | aten::linalg_inv_ex | 5 | 0.261 | 0.0522 | 0.000 |
| 12 | aten::linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 13 | aten::_linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 14 | _SphericalHarmonics | 5 | 0.226 | 0.0452 | 0.226 |
| 15 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.226 | 0.0452 | 0.226 |
| 16 | _FullyFusedProjection | 5 | 0.215 | 0.0430 | 0.215 |
| 17 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.215 | 0.0430 | 0.215 |
| 18 | aten::fill_ | 30 | 0.204 | 0.0068 | 0.204 |
| 19 | aten::zero_ | 25 | 0.189 | 0.0076 | 0.000 |
| 20 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 25 | 0.189 | 0.0076 | 0.189 |

### gsplat_padded_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 5 | 25.770 | 5.1540 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 5 | 25.770 | 5.1540 | 25.388 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 25.388 | 5.0776 | 25.388 |
| 4 | _RasterizeToPixels | 5 | 4.867 | 0.9734 | 4.867 |
| 5 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<32u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 4.867 | 0.9734 | 4.867 |
| 6 | aten::cat | 30 | 2.928 | 0.0976 | 2.928 |
| 7 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 15 | 2.812 | 0.1875 | 2.812 |
| 8 | aten::copy_ | 70 | 2.794 | 0.0399 | 2.794 |
| 9 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 30 | 2.261 | 0.0754 | 2.261 |
| 10 | aten::fill_ | 140 | 1.873 | 0.0134 | 1.873 |
| 11 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 1.833 | 0.0141 | 1.833 |
| 12 | aten::zero_ | 120 | 1.817 | 0.0151 | 0.000 |
| 13 | autograd::engine::evaluate_function: SliceBackward0 | 25 | 1.723 | 0.0689 | 0.000 |
| 14 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 1.549 | 0.0443 | 0.000 |
| 15 | torch::autograd::AccumulateGrad | 35 | 1.549 | 0.0443 | 0.000 |
| 16 | SliceBackward0 | 25 | 1.378 | 0.0551 | 0.000 |
| 17 | aten::slice_backward | 25 | 1.378 | 0.0551 | 0.000 |
| 18 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 1.059 | 0.2118 | 0.000 |
| 19 | _SphericalHarmonicsBackward | 5 | 1.059 | 0.2118 | 0.606 |
| 20 | aten::zeros_like | 45 | 0.940 | 0.0209 | 0.000 |

### gsplat_single_forward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | _RasterizeToPixels | 10 | 6.438 | 0.6438 | 6.438 |
| 2 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.499 | 0.6998 | 3.499 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.939 | 0.5878 | 2.939 |
| 4 | aten::cat | 5 | 1.378 | 0.2756 | 1.378 |
| 5 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.378 | 0.2756 | 1.378 |
| 6 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.800 | 0.0267 | 0.800 |
| 7 | void gsplat::intersect_tile_kernel<float>(bool, unsigned int, unsigned int, unsigned int, long const*, long const*, float const*, int const*, float const*, long const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int*, long*, int*) | 10 | 0.330 | 0.0330 | 0.330 |
| 8 | aten::inverse | 5 | 0.294 | 0.0588 | 0.000 |
| 9 | aten::linalg_inv | 5 | 0.287 | 0.0574 | 0.000 |
| 10 | aten::linalg_inv_ex | 5 | 0.259 | 0.0518 | 0.000 |
| 11 | aten::linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 12 | aten::_linalg_solve_ex | 5 | 0.232 | 0.0464 | 0.000 |
| 13 | _SphericalHarmonics | 5 | 0.225 | 0.0450 | 0.225 |
| 14 | void gsplat::spherical_harmonics_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, glm::vec<3, float, (glm::qualifier)0> const*, float const*, bool const*, float*) | 5 | 0.225 | 0.0450 | 0.225 |
| 15 | aten::mean | 10 | 0.219 | 0.0219 | 0.219 |
| 16 | _FullyFusedProjection | 5 | 0.214 | 0.0428 | 0.214 |
| 17 | void gsplat::projection_ewa_3dgs_fused_fwd_kernel<float>(unsigned int, unsigned int, unsigned int, float const*, float const*, float const*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, float, float, float, float, gsplat::CameraModelType, int*, float*, float*, float*, float*) | 5 | 0.214 | 0.0428 | 0.214 |
| 18 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>) | 10 | 0.209 | 0.0209 | 0.209 |
| 19 | aten::linalg_lu_solve | 5 | 0.142 | 0.0284 | 0.120 |
| 20 | aten::linalg_vector_norm | 5 | 0.120 | 0.0240 | 0.120 |

### gsplat_single_forward_backward

| Rank | Profiler row | Calls | CUDA total ms | CUDA avg ms | Self CUDA ms |
| --- | --- | --- | --- | --- | --- |
| 1 | autograd::engine::evaluate_function: _RasterizeToPixelsBackward | 10 | 20.846 | 2.0846 | 0.000 |
| 2 | _RasterizeToPixelsBackward | 10 | 20.644 | 2.0644 | 20.258 |
| 3 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 12.680 | 2.5360 | 12.680 |
| 4 | void gsplat::rasterize_to_pixels_3dgs_bwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float const*, int const*, float const*, float const*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<2, float, (glm::qualifier)0>*, glm::vec<3, float, (glm::qualifier)0>*, float*, float*) | 5 | 7.578 | 1.5156 | 7.578 |
| 5 | _RasterizeToPixels | 10 | 6.207 | 0.6207 | 6.207 |
| 6 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<16u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 3.453 | 0.6906 | 3.453 |
| 7 | void gsplat::rasterize_to_pixels_3dgs_fwd_kernel<3u, float>(unsigned int, unsigned int, unsigned int, bool, glm::vec<2, float, (glm::qualifier)0> const*, glm::vec<3, float, (glm::qualifier)0> const*, float const*, float const*, float const*, bool const*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int const*, int const*, float*, float*, int*) | 5 | 2.754 | 0.5508 | 2.754 |
| 8 | aten::copy_ | 55 | 1.656 | 0.0301 | 1.656 |
| 9 | aten::cat | 10 | 1.455 | 0.1455 | 1.455 |
| 10 | void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 20 | 1.385 | 0.0693 | 1.385 |
| 11 | void at::native::(anonymous namespace)::CatArrayBatchedCopy<float, unsigned int, 3, 128, 1>(float*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<float, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) | 5 | 1.375 | 0.2750 | 1.375 |
| 12 | aten::fill_ | 140 | 1.330 | 0.0095 | 1.330 |
| 13 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 130 | 1.289 | 0.0099 | 1.289 |
| 14 | aten::zero_ | 120 | 1.275 | 0.0106 | 0.000 |
| 15 | autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 35 | 1.090 | 0.0311 | 0.000 |
| 16 | torch::autograd::AccumulateGrad | 35 | 1.090 | 0.0311 | 0.000 |
| 17 | autograd::engine::evaluate_function: _SphericalHarmonicsBackward | 5 | 1.071 | 0.2142 | 0.000 |
| 18 | _SphericalHarmonicsBackward | 5 | 1.071 | 0.2142 | 0.607 |
| 19 | aten::zeros_like | 65 | 0.933 | 0.0144 | 0.000 |
| 20 | void cub::DeviceRadixSortOnesweepKernel<cub::DeviceRadixSortPolicy<long, int, int>::Policy800, false, long, int, int, int>(int*, int*, int*, int const*, long*, long const*, int*, int const*, int, int, int) | 30 | 0.808 | 0.0269 | 0.808 |