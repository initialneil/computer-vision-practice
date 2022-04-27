/***********************************************************/
/**	\file
	\brief		cuda utils used by Kinect Fusion
	\details	
	\author		Yizhong Zhang
	\date		11/13/2013
*/
/***********************************************************/
#pragma once
#include <stdio.h>
#include <cuda.h>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include "helper_cuda.h"

#ifdef WIN32
#pragma comment(lib, "cudart.lib")
#endif

//#define cudaSafeCall ___cudaSafeCall
//static inline void ___cudaSafeCall(cudaError_t err, std::string msg = "")
//{
//	if (cudaSuccess != err) {
//		printf("CUDA error(%s): %s\n", msg.c_str(), cudaGetErrorString(err));
//		exit(-1);
//	}
//}
#define cudaSafeCall(val) check((val), #val, __FILE__, __LINE__)
#define CudaSafeCall(val) check((val), #val, __FILE__, __LINE__)

static __device__ __host__ __forceinline__ int divUp(int total, int grain)
{
	return (total + grain - 1) / grain; 
}

//The texture collection of a given array
struct CudaTextureSurface 
{
	cudaTextureObject_t texture = 0;
	cudaSurfaceObject_t surface = 0;
	cudaArray_t d_array = 0;
};

#if __CUDA_ARCH__ >= 200
enum { CTA_SIZE = 512, MAX_GRID_SIZE_X = 65536 };
#else
enum { CTA_SIZE = 96, MAX_GRID_SIZE_X = 65536 };
#endif

#ifdef __CUDACC__

template <typename T>
static __device__ __forceinline__ T readTexture(cudaTextureObject_t texture, float x, float y)
{
	return tex2D<T>(texture, x + 0.5f, y + 0.5f);
}

template <typename T>
static __device__ __forceinline__ T readTexture(cudaTextureObject_t texture, float x, float y, float z)
{
	return tex3D<T>(texture, x + 0.5f, y + 0.5f, z + 0.5f);
}

namespace prometheus
{
    namespace device
	{

        struct Block
        {
            static __device__ __forceinline__ unsigned int stride()
            {
                return blockDim.x * blockDim.y * blockDim.z;
            }

            static __device__ __forceinline__ int
                flattenedThreadId()
            {
                return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
            }

            template<int CTA_SIZE, typename T, class BinOp>
            static __device__ __forceinline__ void reduce(volatile T* buffer, BinOp op)
            {
                int tid = flattenedThreadId();
                T val = buffer[tid];

                if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
                if (CTA_SIZE >= 512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
                if (CTA_SIZE >= 256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
                if (CTA_SIZE >= 128) { if (tid < 64) buffer[tid] = val = op(val, buffer[tid + 64]); __syncthreads(); }

                if (tid < 32)
                {
                    if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
                    if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
                    if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
                    if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
                    if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
                    if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
                }
            }

            template<int CTA_SIZE, typename T, class BinOp>
            static __device__ __forceinline__ T reduce(volatile T* buffer, T init, BinOp op)
            {
                int tid = flattenedThreadId();
                T val = buffer[tid] = init;
                __syncthreads();

                if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
                if (CTA_SIZE >= 512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
                if (CTA_SIZE >= 256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
                if (CTA_SIZE >= 128) { if (tid < 64) buffer[tid] = val = op(val, buffer[tid + 64]); __syncthreads(); }

                if (tid < 32)
                {
                    if (CTA_SIZE >= 64) { buffer[tid] = val = op(val, buffer[tid + 32]); }
                    if (CTA_SIZE >= 32) { buffer[tid] = val = op(val, buffer[tid + 16]); }
                    if (CTA_SIZE >= 16) { buffer[tid] = val = op(val, buffer[tid + 8]); }
                    if (CTA_SIZE >= 8) { buffer[tid] = val = op(val, buffer[tid + 4]); }
                    if (CTA_SIZE >= 4) { buffer[tid] = val = op(val, buffer[tid + 2]); }
                    if (CTA_SIZE >= 2) { buffer[tid] = val = op(val, buffer[tid + 1]); }
                }
                __syncthreads();
                return buffer[0];
            }
        };

        struct Warp
        {
            enum
            {
                LOG_WARP_SIZE = 5,
                WARP_SIZE = 1 << LOG_WARP_SIZE,
                STRIDE = WARP_SIZE
            };

            /** \brief Returns the warp lane ID of the calling thread. */
            static __device__ __forceinline__ unsigned int
                laneId()
            {
                unsigned int ret;
                asm("mov.u32 %0, %laneid;" : "=r"(ret));
                return ret;
            }

            static __device__ __forceinline__ unsigned int id()
            {
                int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                return tid >> LOG_WARP_SIZE;
            }

            static __device__ __forceinline__
                int laneMaskLt()
            {
#if (__CUDA_ARCH__ >= 200)
                unsigned int ret;
                asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
                return ret;
#else
                return 0xFFFFFFFF >> (32 - laneId());
#endif
            }

            static __device__ __forceinline__ int binaryExclScan(int ballot_mask)
            {
                return __popc(Warp::laneMaskLt() & ballot_mask);
            }
        };


        struct Emulation
        {
            static __device__ __forceinline__ int
                warp_reduce(volatile int *ptr, const unsigned int tid)
            {
                const unsigned int lane = tid & 31; // index of thread in warp (0..31)        

                if (lane < 16)
                {
                    int partial = ptr[tid];

                    ptr[tid] = partial = partial + ptr[tid + 16];
                    ptr[tid] = partial = partial + ptr[tid + 8];
                    ptr[tid] = partial = partial + ptr[tid + 4];
                    ptr[tid] = partial = partial + ptr[tid + 2];
                    ptr[tid] = partial = partial + ptr[tid + 1];
                }
                return ptr[tid - lane];
            }

            static __forceinline__ __device__ int
                Ballot(int predicate, volatile int* cta_buffer)
            {
#if __CUDA_ARCH__ >= 200
                (void)cta_buffer;
                return __ballot(predicate);
#else
                int tid = Block::flattenedThreadId();
                cta_buffer[tid] = predicate ? (1 << (tid & 31)) : 0;
                return warp_reduce(cta_buffer, tid);
#endif
            }

            static __forceinline__ __device__ bool
                All(int predicate, volatile int* cta_buffer)
            {
#if __CUDA_ARCH__ >= 200
                (void)cta_buffer;
                return __all(predicate);
#else
                int tid = Block::flattenedThreadId();
                cta_buffer[tid] = predicate ? 1 : 0;
                return warp_reduce(cta_buffer, tid) == 32;
#endif
            }
        };
    }

}
#endif
