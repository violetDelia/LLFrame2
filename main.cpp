#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
#include <cmath>
using namespace llframe;
#define QUICK_PRINT(val) std::cout << val << std::endl;

int main() {
    try {
        llframe::memory::Memory<int, device::GPU> memory(1, 0);
        llframe::memory::Memory<float, device::CPU> memory1(1, 0);
        memory.fill(2);
        memory1.copy_form(memory);
        std::cout << memory1.get(0);
        using adpat = llframe::blas::Blas_Adapter<llframe::device::GPU>;

        // memory.fill(5, 5, 3);
        // cudaMemcpy(&a, memory.data(), 4, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < 10; i++) {
        //     std::cout << memory.get(i) << std::endl;
        // }

        // memory.~Memory();
        // std::cout << "1" << std::endl;

        // memory_base.fill(0, {1, 2, 3, 4, 5});
        // llframe::memory::Memory<int, device::GPU> memory_2(10, 0);
        // memory_2.copy_form(memory_base);
        // for (int i = 0; i < 10; i++) { QUICK_PRINT(memory_2.get(i)); }
    } catch (llframe::exception::Exception &e) {
        std::cout << e.what() << std::endl;

    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;

    } catch (...) { std::cout << "other" << std::endl; }
}