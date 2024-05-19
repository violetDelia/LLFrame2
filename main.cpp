#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
#include <cmath>
using namespace llframe;
#define QUICK_PRINT(val) std::cout << val << std::endl;

class A {
public:
    A() {
        count++;
    }
    ~A() {
        count--;
    }
    static inline int count = 0;
};
int main() {
    try {
        llframe::tensor::Tensor<3, float, llframe::device::GPU> tensor(
            2, llframe::shape::make_shape(5, 5, 5), 0);
        llframe::tensor::Tensor<3, float, llframe::device::GPU> tensor1(
            3, llframe::shape::make_shape(5, 5, 5), 0);
        llframe::tensor::Tensor_Operator::add(tensor, tensor1);
        for (int i = 0; i < tensor.count(); i++) {
            std::cout << tensor.memory().get(i) << std::endl;
        }

    } catch (llframe::exception::Exception &e) {
        std::cout << e.what() << std::endl;
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}