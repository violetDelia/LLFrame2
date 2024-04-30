#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
int main() {
    try {
        llframe::allocator::Memory_Pool<llframe::device::GPU>::buffer_list_type;
        int a[10] = {1, 2, 3, 4};
        auto p =
            llframe::allocator::Allocator<int, llframe::device::GPU>::allocate(
                10, 0);
        cudaMemcpy(p.get(), a, sizeof(int) * 10, cudaMemcpyHostToDevice);
        int b[10] = {};
        cudaMemcpy(b, p.get(), sizeof(int) * 10, cudaMemcpyDeviceToHost);
        std::cout << b[1] << b[2] << b[3] << b[4] << std::endl;

    } catch (llframe::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}