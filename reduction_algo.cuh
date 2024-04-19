#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

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