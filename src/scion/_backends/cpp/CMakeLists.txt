cmake_minimum_required(VERSION 3.15)
project(scion_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(_cpp_core bindings.cpp)
target_link_libraries(_cpp_core PRIVATE Eigen3::Eigen)
