# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    batched-static-nnet3-kernels.h
    batched-static-nnet3.h
    batched-threaded-nnet3-cuda-online-pipeline.h
    batched-threaded-nnet3-cuda-pipeline.h
    batched-threaded-nnet3-cuda-pipeline2.h
    cuda-decodable-itf.h
    cuda-decoder-common.h
    cuda-decoder-kernels-utils.h
    cuda-decoder-kernels.h
    cuda-decoder.h
    cuda-fst.h
    cuda-online-pipeline-dynamic-batcher.h
    decodable-cumatrix.h
    thread-pool-light.h
    thread-pool.h
)

if(CUDA_FOUND)
    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
    cuda_compile(CUDA_OBJS
        batched-static-nnet3-kernels.cu
        cuda-decoder-kernels.cu
    )
endif()

add_library(kaldi-cudadecoder
    batched-static-nnet3.cc
    batched-threaded-nnet3-cuda-online-pipeline.cc
    batched-threaded-nnet3-cuda-pipeline.cc
    batched-threaded-nnet3-cuda-pipeline2.cc
    cuda-decoder.cc
    cuda-fst.cc
    cuda-online-pipeline-dynamic-batcher.cc
    decodable-cumatrix.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-cudadecoder PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-cudadecoder PUBLIC
    kaldi-cudamatrix
    kaldi-base
    kaldi-matrix
    kaldi-lat
    kaldi-util
    kaldi-matrix
    kaldi-gmm
    kaldi-fstext
    kaldi-hmm
    kaldi-gmm
    kaldi-transform
    kaldi-tree
    kaldi-online2
    kaldi-nnet3
    kaldi-cudafeat
)


install(TARGETS kaldi-cudadecoder
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/cudadecoder)

