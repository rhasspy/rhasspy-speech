# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    feature-online-batched-cmvn-cuda-kernels.h
    feature-online-batched-cmvn-cuda.h
    feature-online-batched-ivector-cuda-kernels.h
    feature-online-batched-ivector-cuda.h
    feature-online-batched-spectral-cuda-kernels.h
    feature-online-batched-spectral-cuda.h
    feature-online-cmvn-cuda.h
    feature-spectral-cuda.h
    feature-window-cuda.h
    lane-desc.h
    online-batched-feature-pipeline-cuda.h
    online-cuda-feature-pipeline.h
    online-ivector-feature-cuda-kernels.h
    online-ivector-feature-cuda.h
)

if(CUDA_FOUND)
    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
    cuda_compile(CUDA_OBJS
        feature-online-batched-cmvn-cuda-kernels.cu
        feature-online-batched-ivector-cuda-kernels.cu
        feature-online-batched-spectral-cuda-kernels.cu
        feature-online-cmvn-cuda.cu
        feature-spectral-cuda.cu
        feature-window-cuda.cu
        online-ivector-feature-cuda-kernels.cu
    )
endif()

add_library(kaldi-cudafeat
    feature-online-batched-cmvn-cuda.cc
    feature-online-batched-ivector-cuda.cc
    feature-online-batched-spectral-cuda.cc
    online-batched-feature-pipeline-cuda.cc
    online-cuda-feature-pipeline.cc
    online-ivector-feature-cuda.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-cudafeat PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-cudafeat PUBLIC
    kaldi-feat
    kaldi-util
    kaldi-matrix
    kaldi-base
    kaldi-cudamatrix
    kaldi-gmm
    kaldi-ivector
    kaldi-online2
)


install(TARGETS kaldi-cudafeat
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/cudafeat)

