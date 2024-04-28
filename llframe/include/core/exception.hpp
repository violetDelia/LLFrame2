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
#include "core/base_type.hpp"
namespace llframe { inline namespace exception {
/**
 * @brief 异常
 *
 */
class Exception {
public:
    using Self = Exception;
    using size_type = size_t;

public: // 构造函数
    constexpr Exception() noexcept = default;
    constexpr Exception(const Self &other) noexcept = default;
    constexpr Exception(Self &&other) noexcept = default;
    constexpr Exception(const char *message) noexcept : _message(message){};
    constexpr Exception(const char *message, const char *file,
                        const size_type line, const char *func_name) noexcept :
        _message(message) {
        this->add_location(file, line, func_name);
    }
    constexpr Exception(const char *file, const size_type line,
                        const char *func_name) noexcept {
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
    constexpr void add_location(const char *file, const size_type line,
                                const char *func_name) noexcept {
        this->_locations.append("\t");
        this->_locations.append(func_name);
        this->_locations.append(": ");
        this->_locations.append(file);
        this->_locations.append("<");
        this->_locations.append(std::to_string(line));
        this->_locations.append(">\n");
    }

    [[nodiscard]] constexpr virtual std::string what() const noexcept {
        return this->_message + this->_locations;
    }

protected:
    // 异常信息
    std::string _message{"exception!"};
    // 异常传递信息
    std::string _locations{"\n"};
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
 * @brief STD的异常
 *
 */
class STD_Exception : public Exception {
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

}} // namespace llframe::exception
/**
 * @brief 抛出异常
 *
 */
#define __LLFRAME_THROW_EXCEPTION_INFO__(exception, message)                   \
    throw exception(message, std::source_location::current().file_name(),      \
                    std::source_location::current().line(),                    \
                    std::source_location::current().function_name());

/**
 * @brief 抛出异常
 *
 */
#define __LLFRAME_THROW_EXCEPTION__(exception)                                 \
    throw exception(std::source_location::current().file_name(),               \
                    std::source_location::current().line(),                    \
                    std::source_location::current().function_name());
/**
 * @brief 更新异常传递信息
 *
 */
#define __LLFRAME_EXCEPTION_ADD_LOCATION__(exception)                          \
    exception.add_location(std::source_location::current().file_name(),        \
                           std::source_location::current().line(),             \
                           std::source_location::current().function_name());   \
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
#define __LLFRAME_CATCH_UPDATA_EXCEPTION__(Exception)                          \
    catch (Exception & exception) {                                            \
        __LLFRAME_EXCEPTION_ADD_LOCATION__(exception)                          \
    }

/**
 * @brief 捕获到其他异常
 *
 */
#define __LLFRAME_CATCH_OTHER__                                                \
    catch (...) {                                                              \
        __LLFRAME_THROW_EXCEPTION__(llframe::Unknown)                          \
    }

#define __LLFRAME_TRY_CATCH_END__                                              \
    __LLFRAME_TRY_END__                                                        \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Bad_Alloc)                     \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Bad_Index)                     \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Bad_Parameter)                 \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Null_Pointer)                  \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::STD_Exception)                 \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Unhandled)                     \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Unimplement)                   \
    __LLFRAME_CATCH_UPDATA_EXCEPTION__(llframe::Unknown)                       \
    catch (std::exception & e){__LLFRAME_THROW_EXCEPTION_INFO__(               \
        llframe::STD_Exception, e.what())} __LLFRAME_CATCH_OTHER__

#define __THROW_UNIMPLEMENTED__                                                \
    __LLFRAME_THROW_EXCEPTION_INFO__(llframe::Unimplement, "Unimplement!")
#endif //__LLFRAME_EXCEPTION_HPP__