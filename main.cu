#include "reduction_algo.cuh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    // uint64_t size = (uint64_t)2 << (uint64_t)29;

    uint64_t size = 8;
    using datatype = float;

    datatype *input_array = (datatype *)malloc(sizeof(datatype) * size);
    datatype output = 0;
    std::fill(input_array, input_array + size, 1);

    std::cout << std::fixed << "Memory Usage: " << (float)(size * sizeof(datatype)) / 1e9 << " GB" << std::endl;

    const auto start_no_threading = std::chrono::steady_clock::now();
    datatype output_no_threading = host_reduction(input_array, size);
    const auto end_no_threading = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff_no_threading = end_no_threading - start_no_threading;

    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "Result of no threaded host code is: " << output_no_threading << std::endl;
    std::cout << "Time to compute: " << diff_no_threading.count() << std::endl;

    const auto start_threading = std::chrono::steady_clock::now();
    datatype output_threading = host_openmp_reduction(input_array, size);
    const auto end_threading = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff_threading = end_threading - start_threading;

    std::cout << "Result of threaded host code is: " << output_threading << std::endl;
    std::cout << "Time to compute: " << diff_threading.count() << std::endl;

    datatype *d_input, *d_output_kernel1, *d_output_kernel2, *d_output_kernel3, *d_output_kernel4;
    datatype ouput_kernel1, ouput_kernel2, ouput_kernel3, ouput_kernel4;

    cudaMalloc((void **)&d_input, sizeof(datatype) * size);
    cudaMalloc((void **)&d_output_kernel1, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel2, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel3, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel4, sizeof(datatype));

    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel1, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel2, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel3, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel4, &output, sizeof(datatype), cudaMemcpyHostToDevice);

    reduce_kernel1<<<1, 1>>>(d_output_kernel1, d_input, size);
    cudaMemcpy(&ouput_kernel1, d_output_kernel1, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);

    dim3 block_kernel2(1024, 1);
    dim3 grid_kernel2((size + block_kernel2.x - 1) / block_kernel2.x);
    reduce_kernel2<<<block_kernel2, grid_kernel2>>>(d_output_kernel2, d_input, size);
    cudaMemcpy(&ouput_kernel2, d_output_kernel2, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);

    dim3 block_kernel3(1024, 1);
    dim3 grid_kernel3((size + block_kernel3.x - 1) / block_kernel3.x);
    reduce_kernel3<<<block_kernel3, grid_kernel3>>>(d_output_kernel3, d_input, size);
    cudaMemcpy(&ouput_kernel3, d_output_kernel3, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);

    dim3 block_kernel4(1024, 1);
    dim3 grid_kernel4((size + block_kernel3.x - 1) / block_kernel3.x);
    reduce_kernel4<<<block_kernel4, grid_kernel4>>>(d_output_kernel4, d_input, size);
    cudaMemcpy(&ouput_kernel4, d_output_kernel4, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);

    std::cout << "Result of kernel 1 code is: " << ouput_kernel1 << std::endl;
    std::cout << "Result of kernel 2 host code is: " << ouput_kernel2 << std::endl;
    std::cout << "Result of kernel 3 host code is: " << ouput_kernel3 << std::endl;
    std::cout << "Result of kernel 4 host code is: " << ouput_kernel4 << std::endl;

    free(input_array);

    cudaFree(d_input);
    cudaFree(d_output_kernel3);
}