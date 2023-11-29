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

BENCHMARK(BM_HostApi<101>);
BENCHMARK(BM_HostApi<121>);
BENCHMARK(BM_HostApi<141>);
BENCHMARK(BM_HostApi<142>);
BENCHMARK(BM_HostApi<161>);

BENCHMARK_MAIN();
