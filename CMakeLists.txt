include(FetchContent)

cmake_minimum_required(VERSION 4.1)
project(TermedNet)

set(CMAKE_CXX_STANDARD 23)

# Install the Eigen library
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
)


set(SOURCES src/main.cpp src/MLP.cpp src/Dataset.cpp)

add_executable(TermedNet ${SOURCES})

target_compile_options(TermedNet PRIVATE -march=native -O3)

target_include_directories(TermedNet PRIVATE include)

FetchContent_MakeAvailable(Eigen)
target_link_libraries(TermedNet PRIVATE Eigen3::Eigen)


