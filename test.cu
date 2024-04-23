#include <algorithm>
#include <chrono>
#include <execution>
#include <iomanip>
#include <iostream>

int main()
{
    for (int i = 0; i < 31; ++i)
    {
        uint64_t size = (uint64_t)2 << (uint64_t)i;
        using datatype = double;

        std::cout << std::fixed << "Memory Usage: " << (float)(size * sizeof(datatype)) / 1e9 << " GB" << std::endl;
    }
}