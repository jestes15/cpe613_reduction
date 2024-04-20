#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

const uint64_t blockdim = 1024;

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

template <typename _Type> void run_cub_reduce(_Type *in, _Type *out, size_t size)
{
    size_t temp_storage_bytes = 0;
    void *temp_storage = nullptr;
    int init = 0;
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, size, cub::Sum(), init);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, size, cub::Sum(), init);
    cudaDeviceSynchronize();
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
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + 2 * threadIdx.x;

    if (i < size)
    {
        for (uint64_t stride = 1; stride <= blockDim.x; stride *= 2)
        {
            if (threadIdx.x % stride == 0)
            {
                input[i] += input[i + stride];
            }
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
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + threadIdx.x;

    if (i < size)
    {
        for (uint64_t stride = blockDim.x; stride > 0; stride /= 2)
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
    output[0] = input[0];
}

template <typename _Type> __global__ void reduce_kernel5(_Type *output, _Type *input, uint64_t size)
{
    uint64_t segment = 2 * blockDim.x * blockIdx.x;
    uint64_t i = segment + threadIdx.x;

    __shared__ _Type input_s[blockdim];

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
        if (threadIdx.x < stride)
        {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[0] += input_s[threadIdx.x];
    }
}

template <typename _Type, uint64_t course_factor>
__global__ void reduce_kernel6(_Type *output, _Type *input, uint64_t size)
{
    unsigned int segment = course_factor * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ _Type input_s[blockdim];
    _Type sum = 0.0f;
    for (unsigned int c = 0; c < course_factor * 2; ++c)
    {
        if (i + c * blockDim.x < size)
        {
            sum += input[i + c * blockDim.x];
        }
    }
    input_s[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[0] += input_s[0];
    }
}