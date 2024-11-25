# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_library(kaldi-probe INTERFACE)
target_include_directories(kaldi-probe INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/kaldi>
)


install(TARGETS kaldi-probe EXPORT kaldi-targets)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/probe)

