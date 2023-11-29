#include <benchmark/benchmark.h>

#include "rand_gen.h"

template <int RngType>
static void BM_HostApi(benchmark::State& state) {
  cu::HostApiGenerator gen(RngType);
  constexpr int N = 10000;
  gen.Setup(N);
  float* gpu_result = nullptr;
  cu::CudaReserveMemory((void**)&gpu_result, 0, N * sizeof(float));
  for (auto _ : state) {
    gen.Generate(gpu_result, 0.0f, 1.0f);
  }
}

template <int ApiType>
static void BM_DeviceApi(benchmark::State& state) {
  cu::DeviceApiGenerator<ApiType> gen;
  constexpr int N = 10000;
  gen.Setup(N);
  float* gpu_result = nullptr;
  cu::CudaReserveMemory((void**)&gpu_result, 0, N * sizeof(float));
  for (auto _ : state) {
    gen.Generate(gpu_result, 0.0f, 1.0f);
  }
}

BENCHMARK(BM_HostApi<142>);
BENCHMARK(BM_DeviceApi<cu::XORWOW>);
BENCHMARK(BM_DeviceApi<cu::MRG32k3a>);
BENCHMARK(BM_DeviceApi<cu::Philox4>);

BENCHMARK_MAIN();
