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

template <int ApiType>
struct GetStateType {};

template <>
struct GetStateType<XORWOW> {
  using type = curandState;
};

template <>
struct GetStateType<MRG32k3a> {
  using type = curandStateMRG32k3a;
};

template <>
struct GetStateType<Philox4> {
  using type = curandStatePhilox4_32_10_t;
};

template <int ApiType>
__global__ void SetupKernel(typename GetStateType<ApiType>::type* state, size_t num) {
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num) return;
  curand_init(1234, 0, id, state + id);
}

template <int ApiType>
static typename GetStateType<ApiType>::type* g_dev_states;

template <int ApiType>
void DeviceApiGenerator<ApiType>::Setup(std::size_t num) {
  assert((num & 1) == 0);
  InitStream();
  CUDA_CHECK(cudaMalloc((void**)&g_dev_states<ApiType>, num * sizeof(typename GetStateType<ApiType>::type)));
  const unsigned int thd_num = 512;
  const unsigned int blk_num = (num + thd_num - 1) / thd_num;
  SetupKernel<ApiType><<<blk_num, thd_num, 0, g_stream>>>(g_dev_states<ApiType>, num);
  cudaStreamSynchronize(g_stream);
  num_ = num;
}

template <typename StateType>
__global__ void GenerateNormalKernel(StateType* state, float mean, float stddev, size_t num, float* result) {
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num) return;
  auto v = curand_normal(state + id);
  result[id] = v * stddev + mean;
}

template <int ApiType>
void DeviceApiGenerator<ApiType>::Generate(float* dev_ptr, float mean, float stddev) {
  const unsigned int thd_num = 512;
  const unsigned int blk_num = (num_ + thd_num - 1) / thd_num;
  GenerateNormalKernel<<<blk_num, thd_num, 0, g_stream>>>(g_dev_states<ApiType>, mean, stddev, num_, dev_ptr);
  cudaStreamSynchronize(g_stream);
}

template <int ApiType>
void DeviceApiGenerator<ApiType>::Cleanup() {
  CUDA_CHECK(cudaFree(g_dev_states<ApiType>));
}

template class DeviceApiGenerator<XORWOW>;
template class DeviceApiGenerator<MRG32k3a>;
template class DeviceApiGenerator<Philox4>;

}  // namespace cu
