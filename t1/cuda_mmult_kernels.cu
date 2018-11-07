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
   int i = threadIdx.x;
   int k = threadIdx.y;
   
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
      /* TODO: implement tiled matrix multiplication */

}
