cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(CudaTest LANGUAGES CXX CUDA)

find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(rand_gen SHARED rand_gen.cu)
target_compile_options(rand_gen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_compile_features(rand_gen PRIVATE cuda_std_17)
target_link_libraries(rand_gen PRIVATE CUDA::cudart CUDA::curand)

add_executable(curand_test curand_test.cpp)
target_compile_features(curand_test PRIVATE cxx_std_17)
target_link_libraries(curand_test PRIVATE benchmark::benchmark rand_gen)

add_executable(example1 example1.cu)
target_compile_features(example1 PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(example1 PRIVATE rand_gen)
