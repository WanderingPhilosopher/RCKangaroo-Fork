// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include "defs.h"
#include "RCGpuMemoryOptimization.h"

// Performance test kernels for Problem #2 optimizations

__global__ void test_coalesced_memory_access(u64* dst, const u64* src, u32 size)
{
    u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    u32 stride = blockDim.x * gridDim.x;
    
    for (u32 i = tid; i < size; i += stride) {
        coalesced_load_256(dst + i, src + i, i, 4);
    }
}

__global__ void test_shared_memory_optimization(u64* dst, u32 size)
{
    extern __shared__ u64 shared_data[];
    
    // Test bank conflict avoidance
    u32 tid = threadIdx.x;
    u32 bank_conflict_free_idx = avoid_bank_conflict(tid, SHARED_MEM_BANK_SIZE);
    
    // Load data with optimized access pattern
    if (tid < size) {
        shared_data[bank_conflict_free_idx] = tid;
    }
    
    __syncthreads();
    
    // Store back to global memory
    if (tid < size) {
        dst[tid] = shared_data[bank_conflict_free_idx];
    }
}

__global__ void test_memory_bandwidth(u64* dst, const u64* src, u32 size)
{
    bandwidth_optimized_copy(dst, src, size);
}

#if WARP_SPECIALIZATION_ENABLED
__global__ void test_warp_specialization(u64* data, u32 size)
{
    warp_specialized_memory_operations(data, size);
}
#endif

// Host functions for testing

void run_memory_optimization_tests()
{
    const u32 test_size = 1024 * 1024; // 1M elements
    const u32 num_iterations = 100;
    
    // Allocate host memory
    u64* h_src = new u64[test_size];
    u64* h_dst = new u64[test_size];
    
    // Initialize test data
    for (u32 i = 0; i < test_size; i++) {
        h_src[i] = i;
        h_dst[i] = 0;
    }
    
    // Allocate device memory
    u64* d_src, *d_dst;
    cudaMalloc(&d_src, test_size * sizeof(u64));
    cudaMalloc(&d_dst, test_size * sizeof(u64));
    
    // Copy data to device
    cudaMemcpy(d_src, h_src, test_size * sizeof(u64), cudaMemcpyHostToDevice);
    
    // Test 1: Coalesced memory access
    printf("Testing coalesced memory access...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (u32 i = 0; i < num_iterations; i++) {
        test_coalesced_memory_access<<<256, 256>>>(d_dst, d_src, test_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Coalesced access: %.3f ms per iteration\n", milliseconds / num_iterations);
    
    // Test 2: Shared memory optimization
    printf("Testing shared memory optimization...\n");
    cudaEventRecord(start);
    for (u32 i = 0; i < num_iterations; i++) {
        test_shared_memory_optimization<<<256, 256, 1024 * sizeof(u64)>>>(d_dst, test_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory optimization: %.3f ms per iteration\n", milliseconds / num_iterations);
    
    // Test 3: Memory bandwidth optimization
    printf("Testing memory bandwidth optimization...\n");
    cudaEventRecord(start);
    for (u32 i = 0; i < num_iterations; i++) {
        test_memory_bandwidth<<<256, 256>>>(d_dst, d_src, test_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Bandwidth optimization: %.3f ms per iteration\n", milliseconds / num_iterations);
    
    #if WARP_SPECIALIZATION_ENABLED
    // Test 4: Warp specialization
    printf("Testing warp specialization...\n");
    cudaEventRecord(start);
    for (u32 i = 0; i < num_iterations; i++) {
        test_warp_specialization<<<256, 256>>>(d_dst, test_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Warp specialization: %.3f ms per iteration\n", milliseconds / num_iterations);
    #endif
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] h_src;
    delete[] h_dst;
    
    printf("Memory optimization tests completed.\n");
}

// Benchmark function for comparing original vs optimized performance
void benchmark_memory_optimizations()
{
    printf("=== Memory Optimization Benchmark (Problem #2) ===\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Memory Bandwidth: %.1f GB/s\n", 
           (prop.memoryClockRate * 1e-3f) * (prop.memoryBusWidth / 8) * 2);
    
    // Run tests
    run_memory_optimization_tests();
    
    printf("=== Benchmark completed ===\n");
}