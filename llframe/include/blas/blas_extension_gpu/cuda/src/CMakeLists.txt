##---------- serach llfrmae_cuda files ----------##
file(GLOB_RECURSE LLFRAME_CUDA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*cu)

##---------- compile LLframe_Cuda ----------##
if(BUILD_SHARED_LIBS)
    add_library(LLFrame_Cuda SHARED ${LLFRAME_CUDA_FILES})
else()
    add_library(LLFrame_Cuda STATIC ${LLFRAME_CUDA_FILES})
endif()

target_include_directories(LLFrame_Cuda
    PRIVATE
    $<BUILD_INTERFACE:${LLFRAME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${LLFRAME_INSTALL_DIR}>)
    
target_link_libraries(LLFrame_Cuda
    PRIVATE CUDA::cudart CUDA::cublas cudnn
)

set_property(TARGET LLFrame_Cuda PROPERTY VERSION ${LLFRAME_VERSION})
##---------- install LLframe_Cuda ----------##
install(DIRECTORY "${LLFRAME_INCLUDE_DIR}/blas/blas_extension_gpu/cuda/include" DESTINATION "${LLFRAME_INSTALL_INCLUDE_DIR}/blas/blas_extension_gpu/cuda/include")
#将目标安装至指定路径
install(TARGETS LLFrame_Cuda
    EXPORT LLFrame_CudaTargets
    RUNTIME DESTINATION ${LLFRAME_INSTALL_RUNTIME_DIR} #动态库
    LIBRARY DESTINATION ${LLFRAME_INSTALL_LIB_DIR}  #静态库
    ARCHIVE DESTINATION ${LLFRAME_INSTALL_LIB_DIR})
#安装LLFrameTargets.cmake文件
install(EXPORT LLFrame_CudaTargets
    FILE LLFrame_CudaTargets.cmake
    DESTINATION ${LLFRAME_INSTALL_CMAKE_DIR})
