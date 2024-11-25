# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    clusterable-itf.h
    context-dep-itf.h
    decodable-itf.h
    online-feature-itf.h
    optimizable-itf.h
    options-itf.h
)

add_library(kaldi-itf INTERFACE)
target_include_directories(kaldi-itf INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/kaldi>
)


install(TARGETS kaldi-itf EXPORT kaldi-targets)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/itf)

