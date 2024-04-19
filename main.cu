#include "reduction_algo.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <execution>

int main()
{
    uint64_t size = (uint64_t)2 << (uint64_t)29;
    using datatype = uint64_t;

    datatype *input_array = (datatype *)malloc(sizeof(datatype) * size);
    std::fill(std::execution::par, input_array, input_array + size, 1);

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

    std::cout << "Result of hreaded host code is: " << output_threading << std::endl;
    std::cout << "Time to compute: " << diff_threading.count() << std::endl;
}