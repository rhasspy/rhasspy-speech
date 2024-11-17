cmake_minimum_required(VERSION 3.10)
project(phonetisaurus)

add_executable(phonetisaurus
  phonetisaurus.cc util.cc
)
target_include_directories(phonetisaurus PRIVATE utfcpp .)
target_link_libraries(phonetisaurus PRIVATE fst)
