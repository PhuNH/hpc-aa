#include "cuda_mmult_kernels.h"

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 4)
 *  -> Features: none (basic tiled version, using only global memory)
 */
__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n)
{
   int i = blockIdx.x * TILE_SIZE + threadIdx.x;
   int k = blockIdx.y * TILE_SIZE + threadIdx.y;
   
   float Celem = 0;
   
   for(int j=0; j<n; j++) {
      float Aelem = Ad[i*n+j];
      float Belem = Bd[j*n+k];
      Celem += Aelem*Belem;
   }
   
   Cd[i*n+k] += Celem;
}

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 */
__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n)
{
      /* DONE: implement tiled matrix multiplication */
   __shared__ float Ads[TILE_SIZE][TILE_SIZE];
   __shared__ float Bds[TILE_SIZE][TILE_SIZE];
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int x = blockIdx.x * TILE_SIZE + tx;
   int y = blockIdx.y * TILE_SIZE + ty;
   
   float Cval = 0;
   for (int i = 0; i < gridDim.x; i++) {
      Ads[ty][tx] = Ad[y*n + i*TILE_SIZE + tx];
      Bds[ty][tx] = Bd[(i*TILE_SIZE + ty)*n + x];
      __syncthreads();
      for (int k = 0; k < blockDim.x; k++) {
         Cval += Ads[ty][k] * Bds[k][tx];
      }
      __syncthreads();
   }
   Cd[y*n+x] = Cval;
}
