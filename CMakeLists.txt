cmake_minimum_required(VERSION 3.16)
project(onlineml CXX)
set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -Wall")

add_subdirectory(onlineml)
add_subdirectory(third_party)

add_executable(onlineml-train onlineml/onlineml_cli/train.cc)
add_executable(onlineml-evaluate onlineml/onlineml_cli/evaluate.cc)
target_link_libraries(onlineml-train onlineml)
target_link_libraries(onlineml-evaluate onlineml)

install(TARGETS onlineml-train)
install(TARGETS onlineml-evaluate)
