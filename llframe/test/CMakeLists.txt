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
target_link_libraries(llframe_test PRIVATE GTest::gtest LLFrame)