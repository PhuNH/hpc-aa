#define TILE_SIZE 32

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 4)
 *  -> Features: none (basic tiled version, using only global memory)
 */
__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n);

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 */
__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n);

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 *     - coalesced memory access
 */
__global__ void matrixMultKernel_coalesced(float* Ad, float* Bd, float* Cd, int n);

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA kernel
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 5)
 *  -> Features:
 *     - tiled matrix multiplication with use of shared memory
 *     - coalesced memory access
 *     - overlapping loads of subsequent tile pairs (using registers & shared memory)
 */
__global__ void matrixMultKernel_overlap(float* Ad, float* Bd, float* Cd, int n);

