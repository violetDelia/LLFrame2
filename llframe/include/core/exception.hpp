//    Copyright 2023 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/**
 * @brief 异常
 *
 */
#ifndef __LLFRAME_EXCEPTION_HPP__
#define __LLFRAME_EXCEPTION_HPP__
#include <string>
#include <source_location>
#include <exception>
#include "core/base_type.hpp"
namespace llframe ::exception {
/**
 * @brief 异常
 *
 */
class Exception : public std::exception {
public:
    using Self = Exception;
    using size_type = size_t;

public: // 构造函数
    Exception() = default;
    constexpr Exception(const Self &other) = default;
    constexpr Exception(Self &&other) = default;
    explicit Exception(const char *message) : message_(message) {};
    Exception(const char *message, const char *file, const size_type line, const char *func_name) :
        message_(message) {
        this->add_location(file, line, func_name);
    }
    Exception(const char *file, const size_type line, const char *func_name) {
        this->add_location(file, line, func_name);
    }

public:
    /**
     * @brief 添加故障传递信息
     *
     *
     * @param file 文件名
     * @param line 所在行数
     * @param func_name 函数名称
     */
    constexpr void add_location(const char *file, const size_type line, const char *func_name) {
        this->locations_.append("\t");
        this->locations_.append(func_name);
        this->locations_.append(": ");
        this->locations_.append(file);
        this->locations_.append("<");
        this->locations_.append(std::to_string(line));
        this->locations_.append(">\n");
    }

    /**
     * @brief 输出故障信息
     */
    [[nodiscard]] char const *what() const {
        std::string info = std::string(typeid(*this).name()) + '\n' + this->message_ + this->locations_;
        auto size = info.capacity();
        char *str = new char[size];
        memcpy(str, info.data(), size);
        return str;
    }

protected:
    // 异常信息
    std::string message_{"exception!"};
    // 异常传递信息
    std::string locations_{"\n"};
};

/**
 * @brief 未实现导致的异常
 *
 */
class Unimplement : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 分配内存导致的异常
 *
 */
class Bad_Alloc : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 错误索引导致的异常
 *
 */
class Bad_Index : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 空指针导致的异常
 *
 */
class Null_Pointer : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 错误参数导致的异常
 *
 */
class Bad_Parameter : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 无法处理的异常
 *
 */
class Unhandled : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 未知的异常
 *
 */
class Unknown : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 范围错误导致的异常
 *
 */
class Bad_Range : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 调用Cuda导致的异常
 *
 */
class CUDA_Error : public Exception {
public:
    using Exception::Exception;
};

/**
 * @brief 调用cublas的异常
 *
 */
class CuBLAS_Errot : public CUDA_Error {
public:
    using CUDA_Error::CUDA_Error;
};

/**
 * @brief 调用cudnn的异常
 *
 */
class CuDNN_Errot : public CUDA_Error {
public:
    using CUDA_Error::CUDA_Error;
};

} // namespace llframe::exception
/**
 * @brief 抛出异常
 *
 */
#define __LLFRAME_THROW_EXCEPTION_INFO__(exception, message)                                       \
    throw exception(message, std::source_location::current().file_name(),                          \
                    std::source_location::current().line(),                                        \
                    std::source_location::current().function_name());

/**
 * @brief 抛出异常
 *
 */
#define __LLFRAME_THROW_EXCEPTION__(exception)                                                     \
    throw exception(std::source_location::current().file_name(),                                   \
                    std::source_location::current().line(),                                        \
                    std::source_location::current().function_name());
/**
 * @brief 更新异常传递信息
 *
 */
#define __LLFRAME_EXCEPTION_ADD_LOCATION__(exception)                                              \
    exception.add_location(std::source_location::current().file_name(),                            \
                           std::source_location::current().line(),                                 \
                           std::source_location::current().function_name());                       \
    throw exception;

/**
 * @brief try catch 语句开头
 *
 */
#define __LLFRAME_TRY_CATCH_BEGIN__ try {
/**
 * @brief try 语句结尾
 *
 */
#define __LLFRAME_TRY_END__ }

/**
 * @brief 捕获异常并且更新传递信息
 *
 */
#define __LLFRAME_CATCH_UPDATA_EXCEPTION__(Exception)                                              \
    catch (Exception & exception) {                                                                \
        __LLFRAME_EXCEPTION_ADD_LOCATION__(exception)                                              \
    }

/**
 * @brief 捕获到其他异常
 *
 */
#define __LLFRAME_CATCH_OTHER__                                                                    \
    catch (...) {                                                                                  \
        __LLFRAME_THROW_EXCEPTION__(llframe::exception::Unknown)                                   \
    }

#define __LLFRAME_TRY_CATCH_END__                                                                  \
    __LLFRAME_TRY_END__                                                                            \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Bad_Alloc)                              \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Bad_Index)                              \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Bad_Parameter)                          \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Bad_Range)                              \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Null_Pointer)                           \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Unhandled)                              \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Unimplement)                            \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::Unknown)                                \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::CuBLAS_Errot)                           \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::CuDNN_Errot)                            \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::exception::CUDA_Error)
  //__LLFRAME_CATCH_OTHER__

#define __THROW_UNIMPLEMENTED__                                                                    \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Unimplement, "Unimplement!")

#define __LLFRAME_THROW_UNHANDLED__                                                                \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Unhandled, "Unhandled!")

#define __LLFRAME_THROW_UNHANDLED_INFO__(message)                                                  \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Unhandled, message)

#define __LLFRAME_THROW_CUDA_ERROR__                                                               \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CUDA_Error, "cuda error!")

#define __LLFRAME_THROW_CUDA_ERROR_INFO__(cudaError_t)                                             \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CUDA_Error,                               \
                                     std::to_string(cudaError_t).data())
#define __LLFRAME_THROW_CUBLAS_ERROR__                                                             \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CuBLAS_Errot, "cublas error!")

#define __LLFRAME_THROW_CUBLAS_ERROR_INFO__(cublasStatus_t)                                        \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CuBLAS_Errot,                             \
                                     std::to_string(cublasStatus_t).data())
#define __LLFRAME_THROW_CUDNN_ERROR__                                                              \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CuDNN_Errot, "cudnn error!")

#define __LLFRAME_THROW_CUDNN_ERROR_INFO__(cudnnStatus_t)                                          \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::CuDNN_Errot,                              \
                                     std::to_string(cudnnStatus_t).data())

#endif //__LLFRAME_EXCEPTION_HPP__