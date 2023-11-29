#include <benchmark/benchmark.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "rand_gen.h"

static void CheckMeanStd(const std::vector<float>& data, float mean, float stddev) {
  if (data.empty()) {
    std::cout << "data empty" << std::endl;
    return;
  }
  float data_mean = 0.0f;
  for (auto v : data) data_mean += v;
  data_mean /= static_cast<float>(data.size());
  float data_stddev = 0.0f;
  for (auto v : data) {
    float err = v - data_mean;
    data_stddev += err * err;
  }
  data_stddev = std::sqrt(data_stddev / static_cast<float>(data.size()));
  std::cout << "  mean: " << mean << " " << data_mean << " " << (data_mean - mean) << "\n"
            << "stddev: " << stddev << " " << data_stddev << " " << (data_stddev - stddev) / stddev << "\n";
}

template <typename Generator>
static void Validate(std::size_t num, float mean, float stddev) {
  Generator gen;
  gen.Setup(num);
  float* gpu_result = nullptr;
  cu::CudaReserveMemory(reinterpret_cast<void**>(&gpu_result), 0, num * sizeof(float));
  gen.Generate(gpu_result, mean, stddev);
  // std::cout << "1------------------------------\n";
  // cu::InspectValues(gpu_result, num);
  std::vector<float> result(num);
  cu::MemcpyToHost(result.data(), gpu_result, num * sizeof(float));
  CheckMeanStd(result, mean, stddev);
  // clean up
  gen.Cleanup();
  cu::CudaFree(gpu_result);
}

int main(int argc, char** argv) {
  // Validate<cu::HostApiGenerator>(10000, 10.0f, 3.0f);
  //  Validate<cu::DeviceApiGenerator>(4096, 10.0f, 3.0f);
  Validate<cu::DeviceApiGenerator>(10000, 10.0f, 3.0f);
  return 0;

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
