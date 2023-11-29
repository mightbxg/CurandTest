cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(CudaTest LANGUAGES CXX CUDA)

find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(rand_gen SHARED rand_gen.cu)
target_compile_options(rand_gen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_compile_features(rand_gen PRIVATE cuda_std_17)
target_link_libraries(rand_gen PRIVATE CUDA::cudart CUDA::curand)

add_executable(test_rand_gen test_rand_gen.cpp)
target_link_libraries(test_rand_gen PRIVATE rand_gen)

add_executable(example1 example1.cu)
target_compile_features(example1 PRIVATE cuda_std_17)
target_link_libraries(example1 PRIVATE rand_gen)

add_executable(benchmark_hostapi benchmark_hostapi.cpp)
target_link_libraries(benchmark_hostapi benchmark::benchmark rand_gen)

add_executable(benchmark_deviceapi benchmark_deviceapi.cpp)
target_link_libraries(benchmark_deviceapi benchmark::benchmark rand_gen)
