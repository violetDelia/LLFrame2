#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
int main() {
    try {
        int p[10] = {1, 2, 3, 4};
        int *gpu;
        cudaMalloc(&gpu, sizeof(float) * 10);
        cudaMemcpy(gpu, p, sizeof(float) * 10, cudaMemcpyHostToDevice);
        auto res =
            llframe::blas::Blas_Adapter<llframe::device::GPU>::asum(10, gpu, 2);
        std::cout << res << std::endl;

    } catch (llframe::exception::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::exception::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}