#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
using namespace llframe;
int main() {
    try {
        llframe::memory::Memory<int, device::CPU> memory_base(2, 0);
        std::cout << memory_base.size() << std::endl;
    } catch (llframe::exception::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::exception::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}