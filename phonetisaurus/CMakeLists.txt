cmake_minimum_required(VERSION 3.10)
project(phonetisaurus)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

add_executable(phonetisaurus
  phonetisaurus.cc util.cc
)
target_include_directories(phonetisaurus PRIVATE utfcpp .)
target_link_libraries(phonetisaurus PRIVATE fst)
