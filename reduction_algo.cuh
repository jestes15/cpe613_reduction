#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#define BLOCK_DIM 1024

__global__ void print_array(float *input, int size)
{
    printf("MATRIX = [\n");
    for (int i = 0; i < size; ++i)
    {
        printf("%f ", input[i]);
    }
    printf("]\n");
}

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

template <typename _Type> __device__ void print(_Type arr, uint64_t size)
{
    printf("[ ");
    for (int i = 0; i < size; ++i)
        printf("%g ", arr[i]);
    printf("]\n");
}

template <typename _Type> __global__ void reduce_kernel3(_Type *output, _Type *input, uint64_t size)
{
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + 2 * threadIdx.x;

    if (i < size)
    {
        for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
        {
            if (threadIdx.x % stride == 0)
            {
                input[i] += input[i + stride];
            }
            __syncthreads();

            if (i == 0)
                print(input, size);

            __syncthreads();
        }

        if (i % segment == 0 && i > 0)
        {
            atomicAdd(&input[0], input[i]);
        }

        output[0] = input[0];
    }
}

template <typename _Type> __global__ void reduce_kernel4(_Type *output, _Type *input, uint64_t size)
{
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    if (i < size)
    {
        for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
            {
                input[i] += input[i + stride];
            }

            __syncthreads();
        }

        if (i % segment == 0 && i > 0)
        {
            atomicAdd(&input[0], input[i]);
        }
    }
    *output = input[0];
}
