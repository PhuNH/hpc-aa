#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#ifndef M_PI
#define M_PI 3.14159265
#endif

#include "cuda_mmult_kernels.h"

// define macro OUTPUT to print input & output matrix
//#define OUTPUT

// define macro QUERY_DEVICES to print device information
//#define QUERY_DEVICES


void checkCUDAError(const char *msg);

void zeroMatrix(float *A, int n);
void dstMatrix(float *A, int n);
void CPU_matrixMult(float *A, float *B, float *C, int n, int repeats);
void CUDA_matrixMult(float *A, float *B, float *C, int n, int repeats);
void CUDA_freeMatrix(float *Ad);
void printMatrix(const char* name, float *A, int n);
void printDeviceInfo(cudaDeviceProp devProp);

int main(int argc, const char *argv[]) {
   float *A,*B,*C; /* arrays for matrices */
   int n, m; /* n=matrix size, m=repeats */
   
   cudaEvent_t start_timer, stop_timer;
   float gpu_time;
 
#ifdef QUERY_DEVICES
   // Number of CUDA devices
   int devCount;
   cudaGetDeviceCount(&devCount);
   printf("CUDA Device Query...\n");
   printf("There are %d CUDA devices.\n", devCount);

   // Iterate through devices
   for (int i = 0; i < devCount; ++i) {
       // Get device properties
       printf("\nCUDA Device #%d\n", i);
       cudaDeviceProp devProp;
       cudaGetDeviceProperties(&devProp, i);
       printDeviceInfo(devProp);
   }
#endif

   if (argc < 3) {
      printf("Error: please specify matrix size and number of multiplications: \n");
      printf("Usage: %s <size> <repeat> \n", argv[0]);
      exit(1);      
   };
   
   /* read matrix size and number of repeats */

   n = atoi(argv[1]);
   m = atoi(argv[2]);

   if (n % TILE_SIZE != 0) {
      printf("Error: matrix size has to be a multiple of tile size %d \n", TILE_SIZE);
      exit(1);      
   };

   cudaEventCreate(&start_timer);
   cudaEventCreate(&stop_timer);

   printf("Matrix mult. of size %d (%d repeats): \n", n, m);

   /* allocate and initialise matrices in host memory */

   int size = n*n*sizeof(float);

   A = (float *) malloc(size);
   dstMatrix(A,n);
   B = (float *) malloc(size);
   dstMatrix(B,n);
   C = (float *) malloc(size);
   zeroMatrix(C,n);
   
#ifdef OUTPUT
   printMatrix("A",A,n);
   printMatrix("B",B,n); 
#endif

   /* allocate matrices in device memory and transfer matrices from host to device memory */
   
   float *Ad, *Bd, *Cd;
   
   cudaMalloc((void**)&Ad, size); checkCUDAError("allocate memory for A");
   cudaMalloc((void**)&Bd, size); checkCUDAError("allocate memory for B");
   cudaMalloc((void**)&Cd, size); checkCUDAError("allocate memory for C");

   cudaMemcpy(Ad,A, size, cudaMemcpyHostToDevice); checkCUDAError("memory of A not transferred");
   cudaMemcpy(Bd,B, size, cudaMemcpyHostToDevice); checkCUDAError("memory of B not transferred");
   cudaMemcpy(Cd,C, size, cudaMemcpyHostToDevice); checkCUDAError("memory of C not transferred");

   /* perform matrix multiplication (m repeats) */

   cudaEventRecord(start_timer, 0);
      //CPU_matrixMult(A, B, C, n, m);
      CUDA_matrixMult(Ad,Bd,Cd,n,m);
   cudaEventRecord(stop_timer, 0);

   /* transfer result matrix back from device to host memory and deallocate device matrices */

   cudaMemcpy(C,Cd, size, cudaMemcpyDeviceToHost); checkCUDAError("memory of C not transferred back");

   cudaFree(Ad);
   cudaFree(Bd);
   cudaFree(Cd);

#ifdef OUTPUT
   printMatrix("C", C, n);
#endif

   /* deallocate host matrices, print results */

   free(A);
   free(B);
   free(C);
   
   cudaEventSynchronize(stop_timer);
   cudaEventElapsedTime(&gpu_time, start_timer, stop_timer);

   printf("Elapsed time : %.3f s \n", gpu_time / 1000.0f);
   printf("Performance  : %.0f MFlop/s \n", float(m) * (2.0f * n - 1.0f) * n * n / (gpu_time / 1000.0f * 1024.f * 1024.f));

   cudaEventDestroy(start_timer);
   cudaEventDestroy(stop_timer);

   return(0);
}

/* set Matrix values to zero */
void zeroMatrix(float *A, int n)
{
   int i,k;

   for (i=0; i<n; i++)
     for (k=0; k<n; k++)
	    A[i*n+k] = 0;
}

/* initialise Matrix: discrete Sine Transform */
void dstMatrix(float *A, int n)
{
   int i,k;

   for (i=0; i<n; i++)
     for (k=0; k<n; k++)
	    A[i*n+k] = sin( ((i+1)*(k+1)*M_PI)/(n+1));
}



/* 
 * matrix multiplication C += A*B 
 *  -> standard C implementation
 */
void CPU_matrixMult(float *A, float *B, float *C, int n, int repeats) {
	int i,j,k;
    float tmp;

	for(int r=0; r<repeats; r++) {
    	for (i=0; i<n; i++) {
			for (j=0; j<n; j++) {
				tmp = A[i*n+j];

				for (k=0; k<n; k++) {
					C[i*n+k] += tmp * B[j*n+k];
				}
    		}
		}
    }
}

/* 
 * matrix multiplication C += A*B 
 *  -> CUDA implementation: kernel invocation
 *     (implementation adopted from Kirk&Hwu: 
 *      "Programming Massively Parallel Processors, chapter 3)
 */
__host__ void CUDA_matrixMult(float *Ad, float *Bd, float *Cd, int n, int repeats) {
   dim3 dimBlock(TILE_SIZE,TILE_SIZE);
   dim3 dimGrid(n/TILE_SIZE,n/TILE_SIZE);
   
   for(int i=0; i<repeats; i++) {
      // matrixMultKernel_global<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,n);
      // matrixMultKernel_tiled<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,n);
       matrixMultKernel_coalesced<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,n);
      // matrixMultKernel_overlap<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,n);
   }
   checkCUDAError("matrix multiplication kernel failed");
}

/* print Matrix */
void printMatrix(const char* name, float *A, int n)
{
   int i,k;

   printf("Matrix %s (size %d)\n",name,n);

   for (i=0; i<n; i++) {
     for (k=0; k<n; k++) {
       printf("%f ", A[i*n+k]);
     }

     printf("\n");
   }
}

/*
 * helper function to check for errors in CUDA calls
 * source: NVIDIA
 */
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err) {
        fprintf(stderr, "\nCuda error (%s): %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

#ifdef QUERY_DEVICES
// Print device info
void printDeviceInfo(cudaDeviceProp devProp) {
    printf("Revision number:               %d.%d\n", devProp.major, devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu MB\n",  devProp.totalGlobalMem / (1024 * 1024));
    printf("Total shared memory per block: %lu kB\n",  devProp.sharedMemPerBlock / 1024);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu MB\n",  devProp.memPitch / (1024 * 1024));
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);

    printf("Maximum dimensions of block:   %d %d %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Maximum dimensions of grid:    %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

    printf("Clock rate:                    %d MHz\n",  devProp.clockRate / 1000);
    printf("Total constant memory:         %lu kB\n",  devProp.totalConstMem / 1024);
    printf("Texture alignment:             %lu B\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("\n");
}
#endif
