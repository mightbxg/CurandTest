#pragma once

#include <cstdint>

namespace cu {

void CudaFree(void* dev_ptr);

void CudaReserveMemory(void** dev_ptr, std::size_t old_size, std::size_t new_size);

void MemcpyToDevice(void* dev_ptr, const void* host_ptr, std::size_t size);

void MemcpyToHost(void* host_ptr, const void* dev_ptr, std::size_t size);

void InspectValues(float* dev_ptr, std::size_t num);

class Generator {
 public:
  virtual void Setup(std::size_t num) = 0;
  virtual void Generate(float* dev_ptr, float mean, float stddev) = 0;
  virtual void Cleanup() {}

 protected:
  std::size_t num_{0};
};

class HostApiGenerator : public Generator {
 public:
  void Setup(std::size_t num) override;
  void Generate(float* dev_ptr, float mean, float stddev) override;
};

enum DeviceApiType {
  XORWOW,
  MRG32k3a,
  Philox4,
};

template <int ApiType>
class DeviceApiGenerator : public Generator {
 public:
  void Setup(std::size_t num) override;
  void Generate(float* dev_ptr, float mean, float stddev) override;
  void Cleanup() override;
};

}  // namespace cu
