//    Copyright 2023 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/**
 * @brief gpu 算子扩展  vec/vec 头文件
 *
 */

#ifndef LLFRAME_BLAS_BLAS_EXTENSION_GPU_CUDA_DIVIDE_VV_CUH
#define LLFRAME_BLAS_BLAS_EXTENSION_GPU_CUDA_DIVIDE_VV_CUH

namespace llframe::blas::extension::gpu::cuda {
template <class X, class Y>
__global__ void divide_vv_kernel(X *x, Y *y);

template <class X, class Y>
void divide_vv_impl(const int n, X *x, Y *y, const cudaDeviceProp &prop);

extern "C" void divide_vv_i8i8(const int n, signed char *x, signed char *y,
                               const cudaDeviceProp &prop);
extern "C" void divide_vv_i32i32(const int n, signed int *x, signed int *y,
                                 const cudaDeviceProp &prop);
extern "C" void divide_vv_f32f32(const int n, float *x, float *y,
                                 const cudaDeviceProp &prop);
extern "C" void divide_vv_f64f64(const int n, double *x, double *y,
                                 const cudaDeviceProp &prop);
} // namespace llframe::blas::extension::gpu::cuda
#endif // LLFRAME_BLAS_BLAS_EXTENSION_GPU_CUDA_DIVIDE_VV_CUHCUH