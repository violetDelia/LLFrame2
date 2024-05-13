#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
#include <cmath>
using namespace llframe;
#define QUICK_PRINT(val) std::cout << val << std::endl;

int main() {
    try {
        llframe::tensor::Tensor<3,int,llframe::device::GPU> tensor(llframe::shape::make_shape(2, 2, 2),
                                          7, 0);

        std::cout << tensor.get_device_id() << std::endl;
        std::cout << tensor.shape() << std::endl;
        std::cout << tensor.stride() << std::endl;
        auto memory = tensor.memory_ref();

        for (int i = 0; i < memory.size(); i++) { memory.set(i, i); }

        auto memory_other = tensor.memory();
        for (int i = 0; i < memory_other.size(); i++) {
            QUICK_PRINT(memory_other.get(i));
        }

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