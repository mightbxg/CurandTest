#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <mutex>
#include <random>
#include <sstream>

#include "rand_gen.h"

#define CUPRINT(x, ...) \
  { printf("\33[33m(CUDA) " x "\n\33[0m", ##__VA_ARGS__); }

#define CUDA_CHECK(err)                                                          \
  do {                                                                           \
    cudaError_t err_ = (err);                                                    \
    if (err_ != cudaSuccess) {                                                   \
      std::stringstream ss;                                                      \
      ss << "CUDA error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                        \
    }                                                                            \
  } while (false)

#define CURAND_CHECK(err)                                                          \
  do {                                                                             \
    curandStatus_t err_ = (err);                                                   \
    if (err_ != CURAND_STATUS_SUCCESS) {                                           \
      std::stringstream ss;                                                        \
      ss << "cuRAND error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                          \
    }                                                                              \
  } while (false)

#define CUDA_CHECK_LAST_ERROR                                                    \
  do {                                                                           \
    cudaError_t err_ = cudaGetLastError();                                       \
    if (err_ != cudaSuccess) {                                                   \
      std::stringstream ss;                                                      \
      ss << "CUDA error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                        \
    }                                                                            \
  } while (false)

namespace cu {

void CudaFree(void* dev_ptr) { CUDA_CHECK(cudaFree(dev_ptr)); }

void CudaReserveMemory(void** dev_ptr, size_t old_size, size_t new_size) {
  if (new_size > old_size) {
    cudaFree(*dev_ptr);
    CUDA_CHECK(cudaMalloc(dev_ptr, new_size));
  }
}

void MemcpyToDevice(void* dev_ptr, const void* host_ptr, size_t size) {
  if (size > 0) CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

void MemcpyToHost(void* host_ptr, const void* dev_ptr, size_t size) {
  if (size > 0) CUDA_CHECK(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
}

__global__ void PrintValues(float* data, size_t num) {
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num) return;
  printf("%d: %f %p\n", id, data[id], data + id);
}

void InspectValues(float* dev_ptr, std::size_t num) {
  PrintValues<<<1, num>>>(dev_ptr, num);
  cudaDeviceSynchronize();
}

static cudaStream_t g_stream;
static void InitStream() {
  static std::once_flag flag;
  std::call_once(flag, [] { CUDA_CHECK(cudaStreamCreate(&g_stream)); });
}

static curandGenerator_t g_gen;
void HostApiGenerator::Setup(std::size_t num) {
  InitStream();
  std::random_device rd;
  CURAND_CHECK(curandCreateGenerator(&g_gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CHECK(curandSetStream(g_gen, g_stream));
  CURAND_CHECK(curandSetGeneratorOrdering(g_gen, CURAND_ORDERING_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(g_gen, rd()));
  num_ = num;
}

void HostApiGenerator::Generate(float* dev_ptr, float mean, float stddev) {
  CURAND_CHECK(curandGenerateNormal(g_gen, dev_ptr, num_, mean, stddev));
  cudaStreamSynchronize(g_stream);
}

__global__ void SetupKernel(curandState* state, size_t num) {
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num) return;
  curand_init(1234, 0, id, state + id);
}

/// @note num should be multiples of 2
static curandState* g_dev_states;
void DeviceApiGenerator::Setup(std::size_t num) {
  assert((num & 1) == 0);
  InitStream();
  CUDA_CHECK(cudaMalloc((void**)&g_dev_states, num * sizeof(curandState)));
  const unsigned int thd_num = 512;
  const unsigned int blk_num = (num + thd_num - 1) / thd_num;
  SetupKernel<<<blk_num, thd_num, 0, g_stream>>>(g_dev_states, num);
  cudaStreamSynchronize(g_stream);
  num_ = num;
}

__global__ void GenerateNormalKernel(curandState* state, float mean, float stddev, size_t num, float* result) {
  unsigned id = threadIdx.x * 2 + blockIdx.x * blockDim.x;
  if (id >= num) return;
  auto d = curand_normal2(state + id);
  result[id] = d.x * stddev + mean;
  result[id + 1] = d.y * stddev + mean;
}

void DeviceApiGenerator::Generate(float* dev_ptr, float mean, float stddev) {
  const unsigned int thd_num = 512;
  const unsigned int blk_num = (num_ / 2 + thd_num - 1) / thd_num;
  GenerateNormalKernel<<<blk_num, thd_num, 0, g_stream>>>(g_dev_states, mean, stddev, num_, dev_ptr);
  cudaStreamSynchronize(g_stream);
}

void DeviceApiGenerator::Cleanup() { CUDA_CHECK(cudaFree(g_dev_states)); }

}  // namespace cu