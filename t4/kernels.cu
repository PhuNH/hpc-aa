#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

#define TILE_SIZE 64
#define WARP_SIZE 32

extern "C" void CSR_matvec(int N, int nnz, int* start, int* indices, float* data, float* x, float *y, bool bVectorized);
extern "C" void CSR_create(int N, int nnz, int* start, int * indices, float * data , float * x , float * y, int** start_d, int **indices_d, float **data_d, float **x_d, float **y_d);
extern "C" void CSR_kernel(int N, int nnz, int* start_d, int * indices_d, float * data_d , float * x_d , float * y_d, bool bVectorized);
extern "C" void CSR_destroy(int* start_d, int* indices_d, float* data_d, float* x_d, float* y_d);

extern "C" void ELL_create(int N, int num_cols_per_row, int * indices, float * data , float * x , float * y, int **indices_d, float **data_d, float **x_d, float **y_d);
extern "C" void ELL_kernel(int N, int num_cols_per_row , int * indices_d, float * data_d , float * x_d , float * y_d);
extern "C" void ELL_destroy(int* indices_d, float* data_d, float* x_d, float* y_d);

extern "C" void band_create(int N, int num_cols_per_row, float * data , float * x , float * y, float **data_d, float **x_d, float **y_d);
extern "C" void band_kernel(int N, int num_cols_per_row , float * data_d , float * x_d , float * y_d);
extern "C" void band_destroy(float* data_d, float* x_d, float* y_d);

/**
 * Custom CUDA error check wrapper.
 */
#define checkCUDAError() do {                           \
 cudaError_t error = cudaGetLastError();               \
 if (error != cudaSuccess) {                            \
   printf("(CUDA) %s", cudaGetErrorString(error)); \
   printf(" (" __FILE__ ":%d)\n", __LINE__);  \
  }\
} while (0)

/**
 * Cuda kernel for: CSR_s(A)x = y
 */
__global__ void k_csr_mat_vec_mm(const int N, int *start, int* indices, float *data, float *x, float* y) {
	int row = blockDim.x * blockIdx.x + threadIdx.x ;

	if ( row < N ){
		float dot = 0;
		int row_start = start [ row ];
		int row_end = start [ row+1];

		for ( int jj = row_start ; jj < row_end ; jj ++) {
			dot += data [ jj ] * x [ indices [ jj ]];
		}

		y[row] = dot ;
	}
}

/**
 * Cuda kernel for: CSR_v(A)x = y
 */
__global__ void k_csr2_mat_vec_mm(const int N, int *start, int* indices, float *data, float *x, float* y) {
  __shared__ float vals[TILE_SIZE];
   
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;
  int lane = thread_id & (WARP_SIZE - 1);
  int row = warp_id;

  if (row < N) {
    int row_start = start[row];
    int row_end = start[row + 1];

	// compute running sum per thread
	vals[threadIdx.x] = 0;

	for (int jj = row_start + lane; jj < row_end; jj += WARP_SIZE) {
	  vals[threadIdx.x] += data[jj] * x[indices[jj]];
	}

    // parallel reduction in shared memory
    for (int d = WARP_SIZE >> 1; d >= 1; d >>= 1) {
      if (lane < d) vals[threadIdx.x] += vals[threadIdx.x + d];
    }

    // first thread in a warp writes the result
    if (lane == 0) {
      y[row] = vals[threadIdx.x];
    }
  }
}

/**
 * Cuda kernel for: ELL(A)x = y
 */
__global__ void k_ell_mat_vec_mm ( const int N, const int num_cols_per_row , int * indices,
									float * data , float * x , float * y ) {
	int row = blockDim.x * blockIdx.x + threadIdx.x ;

	if ( row < N ){
		float dot = 0;
		for ( int n = 0; n < num_cols_per_row ; n ++){
			int col = indices [ N * n + row ];
			float val = data [ N * n + row ];
			if ( val != 0)
				dot += val * x [ col ];
		}
		y [ row ] = dot ;
	}
}

/**
 * Cuda kernel for: Band(A)x = y
 */
__global__ void band_matvec(int N, int k_max, 
    float* a, float* x, float* y) {

  int i = TILE_SIZE * blockIdx.x + threadIdx.x;

  if (i < N) {
    float dot = 0;

    for (int k = 0; k < 2 * k_max + 1; k++) {
      float val = a[N * k + i];
      int j = i + k - k_max;

      if (val != 0) dot += val * x[j];
    }

    y[i] = dot;
  }
}

/**
 * Perform: CSR(A)x = y
 */
void CSR_matvec(const int N, const int nnz, int* start, int * indices, float * data , float * x , float * y, const bool bVectorized) {
	int *start_d, *indices_d;
	float *data_d, *x_d, *y_d;

	CSR_create(N, nnz, start, indices, data, x, y, &start_d, &indices_d, &data_d, &x_d, &y_d);

	CSR_kernel(N, nnz, start_d, indices_d, data_d, x_d, y_d, bVectorized);

	cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError();

	CSR_destroy(start_d, indices_d, data_d, x_d, y_d);
}


/**
 * Create CSR matrix
 */
void CSR_create(const int N, const int nnz,
		int * start, int * indices, float * data , float * x , float * y, 
		int ** start_d, int ** indices_d, float **data_d, float **x_d, float **y_d) {

	/************************/
	/* copy to device       */
	/************************/

	cudaMalloc((void **) start_d, (N+1) * sizeof(int));
	checkCUDAError();
	cudaMemcpy(*start_d, start, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) indices_d, nnz * sizeof(int));
	checkCUDAError();
	cudaMemcpy(*indices_d, indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) data_d, nnz * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*data_d, data, nnz * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) x_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) y_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*y_d, y, N * sizeof(float) , cudaMemcpyHostToDevice);
	checkCUDAError();
}

/**
 * Perform: CSR(A)x = y
 */
void CSR_kernel(const int N, const int nnz, int * start_d , int * indices_d, float * data_d , float * x_d , float * y_d, const bool bVectorized) {
	if (bVectorized) {
		//#threads = #rows * #threads per row (= N * WARP_SIZE)
		dim3 grid((N * WARP_SIZE + TILE_SIZE - 1)/TILE_SIZE, 1, 1);
		dim3 block(TILE_SIZE, 1, 1);

		k_csr2_mat_vec_mm <<< grid, block >>> (N, start_d, indices_d, data_d, x_d, y_d);
	} else {
		//#threads = #rows (= N)
		dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, 1, 1);
		dim3 block(TILE_SIZE, 1, 1);

		k_csr_mat_vec_mm <<< grid, block >>> (N, start_d, indices_d, data_d, x_d, y_d);
	}

	checkCUDAError();
}

/**
 * Destroy CSR matrix
 */
void CSR_destroy(int* start_d, int* indices_d, float* data_d, float* x_d, float* y_d) {
	cudaFree(start_d);
	cudaFree(indices_d);
	cudaFree(data_d);
	cudaFree(x_d);
	cudaFree(y_d);
}

/**
 * Create band matrix
 */
void band_create(const int N, const int num_cols_per_row,
		float * data , float * x , float * y, 
		float **data_d, float **x_d, float **y_d) {

	cudaMalloc((void **) data_d, N * num_cols_per_row * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*data_d, data, N * num_cols_per_row * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) x_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) y_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();
}


/**
 * Perform: band(A)x = y
 */
void band_kernel(int N, int k_max , float * data_d , float * x_d , float * y_d) {
	//#threads = #rows (= N)
	dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, 1, 1);
	dim3 block(TILE_SIZE, 1, 1);

	band_matvec <<< grid, block >>> (N, k_max, data_d , x_d, y_d);

	checkCUDAError();
}


/**
 * Destroy ELL matrix
 */
void band_destroy(float* data_d, float* x_d, float* y_d) {
	cudaFree(data_d);
	cudaFree(x_d);
	cudaFree(y_d);
}

/**
 * Create ELL matrix
 */
void ELL_create(const int N, const int num_cols_per_row,
		int * indices, float * data , float * x , float * y, 
		int ** indices_d, float **data_d, float **x_d, float **y_d) {

	cudaMalloc((void **) indices_d, N * num_cols_per_row * sizeof(int));
	checkCUDAError();
	cudaMemcpy(*indices_d, indices, N * num_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) data_d, N * num_cols_per_row * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*data_d, data, N * num_cols_per_row * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) x_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();

	cudaMalloc((void **) y_d, N * sizeof(float));
	checkCUDAError();
	cudaMemcpy(*y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError();
}

/**
 * Perform: ELL(A)x = y
 */
void ELL_kernel(int N, int num_cols_per_row , int * indices_d, float * data_d , float * x_d , float * y_d) {
	//round grid size N/TILE_SIZE up
	dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, 1, 1);
	dim3 block(TILE_SIZE, 1, 1);

	k_ell_mat_vec_mm <<< grid, block >>> (N, num_cols_per_row, indices_d, data_d , x_d, y_d);
	checkCUDAError();
}

/**
 * Destroy ELL matrix
 */
void ELL_destroy(int* indices_d, float* data_d, float* x_d, float* y_d) {
	cudaFree(indices_d);
	cudaFree(data_d);
	cudaFree(x_d);
	cudaFree(y_d);
}

