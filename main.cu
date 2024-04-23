#include "reduction_algo.cuh"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    float milliseconds = 0;

    int block_kernel2, grid_kernel2;
    int block_kernel3, grid_kernel3;
    int block_kernel4, grid_kernel4;
    int block_kernel5, grid_kernel5;
    int block_kernel6, grid_kernel6;

    for (int i = 26; i < 27; ++i)
    {
        uint64_t size = (uint64_t)2 << (uint64_t)i;
        using datatype = double;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        datatype *input_array = (datatype *)malloc(sizeof(datatype) * size);
        datatype zero = 0;
        std::fill(input_array, input_array + size, 1.0);

        std::cout << "-------------------------------------------------------------------\n";
        printf("Input Size: %lu (%g GB)\n", size, (size * sizeof(datatype)) / (float)1e9);

        const auto start_no_threading = std::chrono::steady_clock::now();
        datatype output_no_threading = host_reduction(input_array, size);
        const auto end_no_threading = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff_no_threading = end_no_threading - start_no_threading;
        printf("NO THREAD: %g, %f ms\n", output_no_threading,
               std::chrono::duration_cast<std::chrono::nanoseconds>(diff_no_threading).count() /
                   static_cast<float>(1e6));

        const auto start_threading = std::chrono::steady_clock::now();
        datatype output_threading = host_openmp_reduction(input_array, size);
        const auto end_threading = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff_threading = end_threading - start_threading;
        printf("THREAD: %g, %f ms\n", output_threading,
               std::chrono::duration_cast<std::chrono::nanoseconds>(diff_threading).count() / static_cast<float>(1e6));

        datatype *d_input, *d_output_kernel1, *d_output_kernel2, *d_output_kernel3, *d_output_kernel4,
            *d_output_kernel5, *d_output_kernel6, *d_cub_output, *d_thrust_output;
        datatype output_kernel1, output_kernel2, output_kernel3, output_kernel4, output_kernel5, output_kernel6,
            cub_output, thrust_output;

        cudaMalloc((void **)&d_input, sizeof(datatype) * size);
        cudaMalloc((void **)&d_output_kernel1, sizeof(datatype));
        cudaMalloc((void **)&d_output_kernel2, sizeof(datatype));
        cudaMalloc((void **)&d_output_kernel3, sizeof(datatype));
        cudaMalloc((void **)&d_output_kernel4, sizeof(datatype));
        cudaMalloc((void **)&d_output_kernel5, sizeof(datatype));
        cudaMalloc((void **)&d_output_kernel6, sizeof(datatype));
        cudaMalloc((void **)&d_cub_output, sizeof(datatype));
        cudaMalloc((void **)&d_thrust_output, sizeof(datatype));

        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel1, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel2, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel3, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel4, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel5, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_kernel6, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cub_output, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(d_thrust_output, &zero, sizeof(datatype), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        reduce_kernel1<<<1, 1>>>(d_output_kernel1, d_input, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&output_kernel1, d_output_kernel1, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("KERNEL 1: %g, %f ms\n", output_kernel1, milliseconds);

        block_kernel2 = 1024;
        grid_kernel2 = (size + block_kernel2 - 1) / block_kernel2;

        cudaEventRecord(start);
        reduce_kernel2<<<grid_kernel2, block_kernel2>>>(d_output_kernel2, d_input, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&output_kernel2, d_output_kernel2, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("KERNEL 2: %g, %f ms\n", output_kernel2, milliseconds);

        block_kernel3 = 1024;
        grid_kernel3 = (size + block_kernel3 - 1) / block_kernel3;

        cudaEventRecord(start);
        reduce_kernel3<<<grid_kernel3, block_kernel3>>>(d_output_kernel3, d_input, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&output_kernel3, d_output_kernel3, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("KERNEL 3: %g, %f ms\n", output_kernel3, milliseconds);

        block_kernel4 = 1024;
        grid_kernel4 = (size + block_kernel4 - 1) / block_kernel4;

        cudaEventRecord(start);
        reduce_kernel4<<<grid_kernel4, block_kernel4>>>(d_output_kernel4, d_input, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&output_kernel4, d_output_kernel4, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("KERNEL 4: %g, %f ms\n", output_kernel4, milliseconds);

        block_kernel5 = 1024;
        grid_kernel5 = (size + block_kernel5 - 1) / block_kernel5;

        cudaEventRecord(start);
        reduce_kernel5<<<grid_kernel5, block_kernel5>>>(d_output_kernel5, d_input, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&output_kernel5, d_output_kernel5, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("KERNEL 5: %g, %f ms\n", output_kernel5, milliseconds);

        block_kernel6 = 1024;
        grid_kernel6 = (size + block_kernel6 - 1) / block_kernel6;

        for (int course_factor = 1; course_factor < 64; ++course_factor)
        {
            cudaEventRecord(start);
            reduce_kernel6<datatype><<<grid_kernel6, block_kernel6>>>(d_output_kernel6, d_input, size, course_factor);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaMemcpy(&output_kernel6, d_output_kernel6, sizeof(datatype), cudaMemcpyDeviceToHost);
            cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaMemcpy(d_output_kernel6, &zero, sizeof(datatype), cudaMemcpyHostToDevice);
            printf("KERNEL 6: %d, %g, %f ms\n", course_factor, output_kernel6, milliseconds);
        }

        auto cub_time = run_cub_reduce(d_cub_output, d_input, size);
        cudaMemcpy(&cub_output, d_cub_output, sizeof(datatype), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_input, input_array, sizeof(datatype) * size, cudaMemcpyHostToDevice);
        printf("CUB::REDUCE: %g, %f ms\n", cub_output, cub_time);

        auto thrust_time = run_thrust_reduce(d_thrust_output, d_input, size);
        cudaMemcpy(&thrust_output, d_thrust_output, sizeof(datatype), cudaMemcpyDeviceToHost);
        printf("THRUST::REDUCE: %g, %f ms\n", thrust_output, thrust_time);

        free(input_array);

        cudaFree(d_input);
        cudaFree(d_output_kernel1);
        cudaFree(d_output_kernel2);
        cudaFree(d_output_kernel3);
        cudaFree(d_output_kernel4);
        cudaFree(d_output_kernel5);
        cudaFree(d_output_kernel6);
        cudaFree(d_cub_output);

        std::cout << "-------------------------------------------------------------------\n";

        printf("\n\n\n");
    }
}