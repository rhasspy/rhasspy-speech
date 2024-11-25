# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
add_library(kaldi-makefiles INTERFACE)
target_include_directories(kaldi-makefiles INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/kaldi>
)


install(TARGETS kaldi-makefiles EXPORT kaldi-targets)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/makefiles)

