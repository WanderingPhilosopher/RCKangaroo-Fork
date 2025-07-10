// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include "defs.h"

// Memory optimization utilities for Problem #2

// Coalesced memory access patterns
__device__ __forceinline__ void coalesced_load_256(u64* dst, const u64* src, u32 offset, u32 stride)
{
    // Ensure 128-byte aligned access for optimal coalescing
    u32 aligned_offset = (offset * stride) & ~(MEMORY_ALIGNMENT - 1);
    u32 thread_offset = (offset * stride) & (MEMORY_ALIGNMENT - 1);
    
    // Load 256 bits (4 u64) with coalesced access
    u32 base_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    u32 global_idx = aligned_offset + base_idx + thread_offset;
    
    // Use vectorized load for better memory throughput
    *((int4*)dst) = *((int4*)(src + global_idx));
}

__device__ __forceinline__ void coalesced_store_256(u64* dst, const u64* src, u32 offset, u32 stride)
{
    // Ensure 128-byte aligned access for optimal coalescing
    u32 aligned_offset = (offset * stride) & ~(MEMORY_ALIGNMENT - 1);
    u32 thread_offset = (offset * stride) & (MEMORY_ALIGNMENT - 1);
    
    // Store 256 bits (4 u64) with coalesced access
    u32 base_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    u32 global_idx = aligned_offset + base_idx + thread_offset;
    
    // Use vectorized store for better memory throughput
    *((int4*)(dst + global_idx)) = *((int4*)src);
}

// Shared memory bank conflict avoidance
__device__ __forceinline__ u32 avoid_bank_conflict(u32 index, u32 bank_size)
{
    // Pad shared memory access to avoid bank conflicts
    u32 bank = index % bank_size;
    u32 row = index / bank_size;
    return row * (bank_size + 1) + bank;
}

// Optimized shared memory allocation with bank conflict avoidance
__device__ __forceinline__ void* get_shared_memory_ptr(u32 offset, u32 size)
{
    extern __shared__ u8 shared_memory[];
    
    // Align to 128-byte boundary for optimal access
    u32 aligned_offset = (offset + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
    
    // Avoid bank conflicts by padding
    if (size <= SHARED_MEM_BANK_SIZE) {
        aligned_offset = avoid_bank_conflict(aligned_offset, SHARED_MEM_BANK_SIZE);
    }
    
    return (void*)(shared_memory + aligned_offset);
}

// Asynchronous memory operations for Ampere/Hopper
#if ASYNC_MEMORY_ENABLED
#include <cuda/pipeline>
#include <cuda/barrier>

__device__ __forceinline__ void async_memory_copy(u64* dst, const u64* src, u32 size, 
                                                 cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // Use async copy for better memory throughput
    cuda::memcpy_async(dst, src, size, pipe);
}

__device__ __forceinline__ void async_memory_copy_2d(u64* dst, const u64* src, 
                                                    u32 width, u32 height, u32 pitch,
                                                    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // 2D async copy for structured data
    cuda::memcpy_2d_async(dst, pitch, src, pitch, width, height, pipe);
}
#endif

// Memory prefetching for better cache utilization
__device__ __forceinline__ void prefetch_global_memory(const u64* ptr, u32 size)
{
    // Prefetch data into L2 cache
    #pragma unroll
    for (u32 i = 0; i < size; i += 64) {
        __prefetch(ptr + i);
    }
}

// Optimized memory access patterns
__device__ __forceinline__ void optimized_memory_access(u64* dst, const u64* src, 
                                                       u32 pattern, u32 count)
{
    switch (pattern) {
        case MEMORY_ACCESS_PATTERN_COALESCED:
            // Coalesced access pattern
            for (u32 i = 0; i < count; i += 4) {
                coalesced_load_256(dst + i, src + i, i, 4);
            }
            break;
            
        case MEMORY_ACCESS_PATTERN_STRIDED:
            // Strided access pattern with prefetching
            for (u32 i = 0; i < count; i += 16) {
                prefetch_global_memory(src + i, 64);
                *((int4*)(dst + i)) = *((int4*)(src + i));
            }
            break;
            
        case MEMORY_ACCESS_PATTERN_RANDOM:
            // Random access pattern with cache optimization
            for (u32 i = 0; i < count; i++) {
                u32 idx = (i * 1103515245 + 12345) % count;  // Simple hash for random access
                dst[i] = src[idx];
            }
            break;
    }
}

// Warp specialization for RTX 40xx (Problem #2 optimization)
#if WARP_SPECIALIZATION_ENABLED
__device__ __forceinline__ void warp_specialized_memory_operations(u64* data, u32 size)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    switch (warp_id) {
        case 0: case 1:
            // Warps 0-1: Memory-intensive operations with coalesced access
            if (lane_id < 16) {
                coalesced_load_256(data + lane_id * 4, data + lane_id * 4, lane_id, 4);
            }
            break;
            
        case 2: case 3: case 4: case 5:
            // Warps 2-5: Arithmetic operations with optimized memory access
            if (lane_id < 8) {
                optimized_memory_access(data + lane_id * 8, data + lane_id * 8, 
                                      MEMORY_ACCESS_PATTERN_COALESCED, 32);
            }
            break;
            
        case 6: case 7:
            // Warps 6-7: Coordination and synchronization with minimal memory access
            if (lane_id == 0) {
                // Only first thread in warp performs coordination
                __syncwarp();
            }
            break;
    }
}
#endif

// Memory pool for efficient allocation
struct OptimizedMemoryPool {
    __device__ __forceinline__ void* allocate(u32 size) {
        extern __shared__ u8 shared_memory[];
        static __shared__ u32 pool_offset;
        
        if (threadIdx.x == 0) {
            u32 aligned_size = (size + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
            u32 offset = atomicAdd(&pool_offset, aligned_size);
            return (void*)(shared_memory + offset);
        }
        return nullptr;
    }
    
    __device__ __forceinline__ void reset() {
        if (threadIdx.x == 0) {
            extern __shared__ u32 pool_offset;
            pool_offset = 0;
        }
        __syncthreads();
    }
};

// Memory access optimization for L2 cache
__device__ __forceinline__ void l2_cache_optimized_access(u64* dst, const u64* src, u32 size)
{
    // Optimize for L2 cache line size (128 bytes)
    u32 cache_line_size = 128;
    u32 cache_lines = (size + cache_line_size - 1) / cache_line_size;
    
    for (u32 i = 0; i < cache_lines; i++) {
        u32 offset = i * cache_line_size;
        u32 line_size = min(cache_line_size, size - offset);
        
        // Access data in cache-line aligned chunks
        coalesced_load_256(dst + offset, src + offset, offset / 8, 1);
    }
}

// Memory bandwidth optimization
__device__ __forceinline__ void bandwidth_optimized_copy(u64* dst, const u64* src, u32 count)
{
    // Use maximum memory bandwidth by accessing multiple memory channels
    u32 threads_per_warp = 32;
    u32 warps_per_block = blockDim.x / threads_per_warp;
    u32 warp_id = threadIdx.x / threads_per_warp;
    u32 lane_id = threadIdx.x % threads_per_warp;
    
    // Distribute work across memory channels
    for (u32 i = warp_id; i < count; i += warps_per_block) {
        if (lane_id < 4) {  // 4 u64 per thread
            u32 idx = i * 4 + lane_id;
            if (idx < count) {
                dst[idx] = src[idx];
            }
        }
    }
}