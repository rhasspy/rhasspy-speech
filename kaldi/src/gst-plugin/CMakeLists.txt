# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    gst-audio-source.h
    gst-online-gmm-decode-faster.h
)

add_library(kaldi-gst-plugin
    gst-audio-source.cc
    gst-online-gmm-decode-faster.cc
)

target_include_directories(kaldi-gst-plugin PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)


install(TARGETS kaldi-gst-plugin
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/gst-plugin)

