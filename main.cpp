#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
int main() {
    try {
        auto &ins = llframe::Device_Platfrom<llframe::GPU>::get_instance();
        std::cout << "get_instance success" << std::endl;
        auto &device = llframe::Device_Platfrom<llframe::GPU>::get_device(0);
        std::cout << device.property().maxGridSize << std::endl;
        std::cout << "get_device success" << std::endl;
    } catch (llframe::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}