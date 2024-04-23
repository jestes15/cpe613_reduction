#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cstdint>

#define BLOCK_DIM 1024

template <typename _Type> _Type host_reduction(_Type *input, uint64_t size)
{
    _Type output = 0;

    for (uint64_t i = 0; i < size; ++i)
        output += input[i];

    return output;
}

template <typename _Type> _Type host_openmp_reduction(_Type *input, uint64_t size)
{
    _Type output = 0;

#pragma omp parallel for shared(input) reduction(+ : output)
    for (uint64_t i = 0; i < size; ++i)
        output += input[i];

    return output;
}

template <typename _Type> float run_cub_reduce(_Type *output, _Type *input, uint64_t size)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t temp_storage_bytes = 0;
    void *temp_storage = nullptr;
    int init = 0;
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, input, output, size, cub::Sum(), init);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    cudaEventRecord(start);
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, input, output, size, cub::Sum(), init);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(temp_storage);

    return milliseconds;
}

template <typename _Type> float run_thrust_reduce(_Type *output, _Type *input, uint64_t size)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    thrust::device_ptr<_Type> input_thrust_ptr = thrust::device_pointer_cast(input);
    thrust::device_ptr<_Type> output_thrust_ptr = thrust::device_pointer_cast(output);

    cudaEventRecord(start);
    *output_thrust_ptr = thrust::reduce(input_thrust_ptr, input_thrust_ptr + size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

template <typename _Type> __global__ void reduce_kernel1(_Type *output, _Type *input, uint64_t size)
{
    _Type sum = 0;
    for (uint64_t i = 0; i < size; ++i)
        sum += input[i];

    *output = sum;
}

template <typename _Type> __global__ void reduce_kernel2(_Type *output, _Type *input, uint64_t size)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        atomicAdd(output, input[i]);
}

template <typename _Type> __global__ void reduce_kernel3(_Type *output, _Type *input, uint64_t size)
{
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + 2 * threadIdx.x;

    for (uint64_t stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        if (threadIdx.x % stride == 0)
        {
            if (i + stride < size && i < size)
                input[i] += input[i + stride];
        }
    }

    if (threadIdx.x == 0 && i < size)
    {
        atomicAdd(&output[0], input[i]);
    }
}

template <typename _Type> __global__ void reduce_kernel4(_Type *output, _Type *input, uint64_t size)
{
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + threadIdx.x;

    for (uint64_t stride = blockDim.x; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride && i < size && i + stride < size)
        {
            input[i] += input[i + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0 && i < size)
    {
        atomicAdd(&output[0], input[i]);
    }
}

template <typename _Type> __global__ void reduce_kernel5(_Type *output, _Type *input, uint64_t size)
{
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + threadIdx.x;

    __shared__ _Type input_s[BLOCK_DIM];

    if (i < size)
    {
        input_s[threadIdx.x] = input[i];
        if (i + blockDim.x < size)
        {
            input_s[threadIdx.x] += input[i + blockDim.x];
        }
    }
    else
    {
        input_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for (uint64_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x)
        {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&output[0], input_s[threadIdx.x]);
    }
}

template <typename _Type>
__global__ void reduce_kernel6(_Type *output, _Type *input, uint64_t size, uint64_t coarse_factor)
{
    unsigned int segment = coarse_factor * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ _Type input_s[BLOCK_DIM];

    _Type sum;

    if (i < size)
        sum = input[i];
    else
        sum = 0;

    __syncthreads();

    for (unsigned int tile = 1; tile < coarse_factor * 2; ++tile)
    {
        if (i + tile * BLOCK_DIM < size)
        {
            sum += input[i + tile * BLOCK_DIM];
        }
    }

    __syncthreads();
    if (threadIdx.x < BLOCK_DIM)
        input_s[threadIdx.x] = sum;

    for (unsigned int stride = (blockDim.x >> 1); stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride && threadIdx.x + stride < BLOCK_DIM)
        {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(&output[0], input_s[0]);
    }
}