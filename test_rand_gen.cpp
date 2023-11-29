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
  float mean_diff = data_mean - mean;
  float stddev_diff = (data_stddev - stddev) / stddev;
  if (std::abs(mean_diff) > 0.05) std::cout << "\33[33m";
  if (std::abs(mean_diff) > 0.15) std::cout << "\33[31m";
  std::cout << "  mean: " << mean << " " << data_mean << " " << mean_diff << "\n\33[0m";
  if (std::abs(stddev_diff) > 0.02) std::cout << "\33[33m";
  if (std::abs(stddev_diff) > 0.1) std::cout << "\33[31m";
  std::cout << "stddev: " << stddev << " " << data_stddev << " " << stddev_diff << "\n\33[0m";
}

template <typename Generator>
static void Validate(std::size_t num, float mean, float stddev) {
  Generator gen;
  gen.Setup(num);
  float* gpu_result = nullptr;
  cu::CudaReserveMemory(reinterpret_cast<void**>(&gpu_result), 0, num * sizeof(float));
  gen.Generate(gpu_result, mean, stddev);
  std::vector<float> result(num);
  cu::MemcpyToHost(result.data(), gpu_result, num * sizeof(float));
  CheckMeanStd(result, mean, stddev);
  // clean up
  gen.Cleanup();
  cu::CudaFree(gpu_result);
}

int main(int argc, char** argv) {
  std::vector<size_t> nums = {100, 1000, 10000};
  std::vector<float> stddevs = {0.01, 1.0, 10.0};
  float mean = 10.0f;
  for (auto num : nums) {
    for (auto stddev : stddevs) {
      std::cout << "\33[35m---------------------num[" << num << "] mean[" << mean << "] std[" << stddev << "]\33[0m\n";
      Validate<cu::HostApiGenerator>(num, mean, stddev);
      Validate<cu::DeviceApiGenerator<cu::XORWOW>>(num, mean, stddev);
      Validate<cu::DeviceApiGenerator<cu::MRG32k3a>>(num, mean, stddev);
      Validate<cu::DeviceApiGenerator<cu::Philox4>>(num, mean, stddev);
    }
  }
  return 0;
}
