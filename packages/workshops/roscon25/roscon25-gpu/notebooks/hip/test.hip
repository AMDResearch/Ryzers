#include <iostream>
#include <hip/hip_runtime.h>

int main (void) {
    // Detect platform
    hipDeviceProp_t dProperties;
    auto err = hipGetDeviceProperties(&dProperties, 0);
    
    if(err!=HIP_SUCCESS) {
        return -1;
    }
    std::cout << "Hello ROSCON" << std::endl;
    std::cout << "GPU detected: " << dProperties.name << std::endl;
    std::cout << "Total memory: " << dProperties.totalGlobalMem /1024 /1024 << " MB" << std::endl;
    return 0;
}