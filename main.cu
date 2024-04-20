#include "reduction_algo.cuh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    // uint64_t size = (uint64_t)2 << (uint64_t)29;

    uint64_t size = 16384;
    using datatype = float;

    datatype *input_array = (datatype *)malloc(sizeof(datatype) * size);
    datatype output = 0;
    std::fill(input_array, input_array + size, 1.0);

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

    datatype *d_input, *d_output_kernel1, *d_output_kernel2, *d_output_kernel3, *d_output_kernel4, *d_output_kernel5, *d_output_kernel6, *d_cub_output;
    datatype output_kernel1, output_kernel2, output_kernel3, output_kernel4, output_kernel5, output_kernel6, cub_output;

    cudaMalloc((void **)&d_input, sizeof(datatype) * size);
    cudaMalloc((void **)&d_output_kernel1, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel2, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel3, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel4, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel5, sizeof(datatype));
    cudaMalloc((void **)&d_output_kernel6, sizeof(datatype));
    cudaMalloc((void **)&d_cub_output, sizeof(datatype));

    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel1, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel2, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel3, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel4, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel5, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_kernel6, &output, sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cub_output, &output, sizeof(datatype), cudaMemcpyHostToDevice);

    reduce_kernel1<<<1, 1>>>(d_output_kernel1, d_input, size);
    cudaMemcpy(&output_kernel1, d_output_kernel1, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 1 code is: " << output_kernel1 << std::endl;

    dim3 block_kernel2(1024, 1);
    dim3 grid_kernel2((size + block_kernel2.x - 1) / block_kernel2.x);
    reduce_kernel2<<<block_kernel2, grid_kernel2>>>(d_output_kernel2, d_input, size);
    cudaMemcpy(&output_kernel2, d_output_kernel2, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 2 host code is: " << output_kernel2 << std::endl;

    dim3 block_kernel3(1024, 1);
    dim3 grid_kernel3((size + block_kernel3.x - 1) / block_kernel3.x);
    reduce_kernel3<<<block_kernel3, grid_kernel3>>>(d_output_kernel3, d_input, size);
    cudaMemcpy(&output_kernel3, d_output_kernel3, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 3 host code is: " << output_kernel3 << std::endl;

    dim3 block_kernel4(1024, 1);
    dim3 grid_kernel4((size + block_kernel4.x - 1) / block_kernel4.x);
    reduce_kernel4<<<block_kernel4, grid_kernel4>>>(d_output_kernel4, d_input, size);
    cudaMemcpy(&output_kernel4, d_output_kernel4, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 4 host code is: " << output_kernel4 << std::endl;

    dim3 block_kernel5(1024, 1);
    dim3 grid_kernel5((size + block_kernel5.x - 1) / block_kernel5.x);
    reduce_kernel5<<<block_kernel5, grid_kernel5>>>(d_output_kernel5, d_input, size);
    cudaMemcpy(&output_kernel5, d_output_kernel5, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 5 host code is: " << output_kernel5 << std::endl;

    dim3 block_kernel6(1024, 1);
    dim3 grid_kernel6((size + block_kernel6.x - 1) / block_kernel6.x);
    reduce_kernel6<float, 2><<<block_kernel6, grid_kernel6>>>(d_output_kernel6, d_input, size);
    cudaMemcpy(&output_kernel6, d_output_kernel6, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of kernel 6 host code is: " << output_kernel6 << std::endl;

    run_cub_reduce(d_input, d_cub_output, size);
    cudaMemcpy(&cub_output, d_cub_output, sizeof(datatype), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
    std::cout << "Result of cub code is: " << cub_output << std::endl;

    free(input_array);

	cudaFree(d_input);
	cudaFree(d_output_kernel1);
	cudaFree(d_output_kernel2);
	cudaFree(d_output_kernel3);
	cudaFree(d_output_kernel4);
	cudaFree(d_output_kernel5);
    cudaFree(d_output_kernel6);
    cudaFree(d_cub_output);
}