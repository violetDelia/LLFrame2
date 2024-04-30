#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
int main() {
    try {
        llframe::device::GPU(2);

    } catch (llframe::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}