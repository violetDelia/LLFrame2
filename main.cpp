#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
int main() {
    try {
        float a[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float x[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float y[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float *gpu_a;
        float *gpu_x;
        float *gpu_y;
        cudaMalloc(&gpu_x, sizeof(float) * 16);
        cudaMemcpy(gpu_x, x, sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_y, sizeof(float) * 16);
        cudaMemcpy(gpu_y, y, sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_a, sizeof(float) * 16);
        cudaMemcpy(gpu_a, a, sizeof(float) * 16, cudaMemcpyHostToDevice);
        llframe::blas::Blas_Adapter<llframe::device::GPU>::gemm(
            llframe::blas::Blas_Layout::Col_Major,
            llframe::blas::Blas_Transpose::NoTrans,
            llframe::blas::Blas_Transpose::NoTrans, 1, 3, 3, 1, gpu_a, 3, gpu_x,
            4, 0, gpu_y, 5);
        cudaMemcpy(y, gpu_y, sizeof(float) * 16, cudaMemcpyDeviceToHost);
        std::cout << y[0] << " " << y[1] << " " << y[2] << " " << y[3] << " "
                  << y[4] << " " << y[5] << " " << y[6] << " " << y[7] << " "
                  << y[8] << " " << y[9] << " " << y[10] << " " << y[11] << " "
                  << y[2] << std::endl;
        // a [1,1,1,1]
        // b [1,1,1,1]
        //   [1,1,1,1]
        //   [1,1,1,1]
        //   [1,1,1,1]
        llframe::blas::Blas_Adapter<llframe::device::CPU>::gemm(
            llframe::blas::Blas_Layout::Col_Major,
            llframe::blas::Blas_Transpose::NoTrans,
            llframe::blas::Blas_Transpose::NoTrans, 1, 3, 3, 1, a, 3, x, 4, 0,
            y, 5);
        std::cout << y[0] << " " << y[1] << " " << y[2] << " " << y[3] << " "
                  << y[4] << " " << y[5] << " " << y[6] << " " << y[7] << " "
                  << y[8] << " " << y[9] << " " << y[10] << " " << y[11] << " "
                  << y[2] << std::endl;
    } catch (llframe::exception::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::exception::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}