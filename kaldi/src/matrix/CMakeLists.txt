# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    cblas-wrappers.h
    compressed-matrix.h
    jama-eig.h
    jama-svd.h
    kaldi-blas.h
    kaldi-matrix-inl.h
    kaldi-matrix.h
    kaldi-vector-inl.h
    kaldi-vector.h
    matrix-common.h
    matrix-functions-inl.h
    matrix-functions.h
    matrix-lib.h
    numpy-array.h
    optimization.h
    packed-matrix.h
    sp-matrix-inl.h
    sp-matrix.h
    sparse-matrix.h
    srfft.h
    tp-matrix.h
)

add_library(kaldi-matrix
    compressed-matrix.cc
    kaldi-matrix.cc
    kaldi-vector.cc
    matrix-functions.cc
    numpy-array.cc
    optimization.cc
    packed-matrix.cc
    qr.cc
    sp-matrix.cc
    sparse-matrix.cc
    srfft.cc
    tp-matrix.cc
)

target_include_directories(kaldi-matrix PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-matrix PUBLIC
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME matrix-lib-speed-test SOURCES matrix-lib-speed-test.cc DEPENDS kaldi-matrix)
    add_kaldi_test_executable(NAME matrix-lib-test SOURCES matrix-lib-test.cc DEPENDS kaldi-matrix)
    add_kaldi_test_executable(NAME numpy-array-test SOURCES numpy-array-test.cc DEPENDS kaldi-matrix)
    add_kaldi_test_executable(NAME sparse-matrix-test SOURCES sparse-matrix-test.cc DEPENDS kaldi-matrix)
endif()

install(TARGETS kaldi-matrix
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/matrix)

