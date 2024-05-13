##---------- serach test files ----------##
file(GLOB_RECURSE LLFRAME_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*cpp ${CMAKE_CURRENT_SOURCE_DIR}/*hpp)

##---------- llframe_test ----------##
add_executable(llframe_test ${LLFRAME_TEST_FILES})
target_include_directories(
    llframe_test 
    PRIVATE
    $<BUILD_INTERFACE: ${LLFRAME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${LLFRAME_INCLUDE_DIR}>
)
target_link_libraries(llframe_test PUBLIC GTest::gtest LLFrame)
#测试函数有很多数值转换的警告
target_compile_options(llframe_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4244>)