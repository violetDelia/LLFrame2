
#ifndef __LLFRAME_TEST_COMMON__
#define __LLFRAME_TEST_COMMON__
#include "core/base_type.hpp"
#include "core/exception.hpp"
#include "device/device_impl.hpp"
using Exception_Tuple =
    std::tuple<llframe::exception::Bad_Alloc, llframe::exception::Bad_Parameter,
               llframe::exception::Bad_Index, llframe::exception::Exception,
               llframe::exception::Null_Pointer,
               llframe::exception::STD_Exception, llframe::exception::Unhandled,
               llframe::exception::Unimplement, llframe::exception::Unknown,
               llframe::exception::Bad_Range, llframe::exception::CUDA_Error,
               llframe::exception::CuBLAS_Errot,
               llframe::exception::CuDNN_Errot>;

using Device_Tuple = std::tuple<llframe::device::CPU, llframe::device::GPU>
#endif //__LLFRAME_TEST_COMMON__