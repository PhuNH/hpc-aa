#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

#define TILE_SIZE 512
#define WARP_SIZE 32

extern "C" void CSRmatvecmult(int* ptr, int* J, float* Val, int N, int nnz, float* x, float *y, bool bVectorized);
extern "C" void ELLmatvecmult(int N, int num_cols_per_row , int * indices, float * data , float * x , float * y);

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
__global__ void k_csr_mat_vec_mm(int *ptr, int* indices, float *data, int num_rows, float *x, float* y) {
    //TODO: implement the CSR kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        float temp = 0;
        int start = ptr[i], end = ptr[i+1]; // ptr: coalesced
        for (int k = start; k < end; k++) {
            temp += data[k] * x[indices[k]]; // x, data, indices: uncoalesced
        }
        y[i] = temp; // y: coalesced
    }
}

/**
 * Cuda kernel for: CSR_v(A)x = y
 */
__global__ void k_csr2_mat_vec_mm(int *ptr, int* indices, float *data, int num_rows, float *x, float* y) {
    //TODO: implement the vectorized csr kernel
    /*
     * i) Set grid and block size in the kernel call accordingly
     * ii) Assign a matrix row to each warp now
     * iii) Allocate a shared array vals[] for the partial results of a block
     * iv) Compute one row × vector product in a loop. This time, parallelize the loop over all 32 threads in the warp. Take care that access to the arrays indices and data is coalesced.
     * v) Use a reduction of some kind (ideally: binary fan-in) to add up the partial sums in vals[] and add the output to the result vector.
     */
    /* no vectorization
     * usroads.mtx
     * real    0m2.548s
     * user    0m0.272s
     * sys     0m2.212s
     */
    __shared__ float vals[WARP_SIZE][WARP_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row_in_grid = blockIdx.y * blockDim.y + ty;
    if (row_in_grid < num_rows) {
        int k;
        int start = ptr[row_in_grid], end = ptr[row_in_grid+1];
        float temp = 0;
        for (k = tx + start; k < end; k += WARP_SIZE) {
            temp += data[k] * x[indices[k]];
        }
        vals[ty][tx] = temp;
        //__syncthreads();
        
        // First attempt
        /* vectorized, sum by loop
         * usroads.mtx
         * real    0m2.781s
         * user    0m0.288s
         * sys     0m1.683s
         */
        /*int l, p;
        for (k = 1; k < 6; k++) {
            p = (int) powf(2, k-1);
            for (l = 0; l < WARP_SIZE/(2*p); l++) {
                vals[ty][2*p*l] += vals[ty][2*p*l+p];
            }
        }*/
        // Second attempt
        /* vectorized, sum by binary fan-in
         * usroads.mtx
         * real    0m2.627s
         * user    0m0.212s
         * sys     0m1.543s
         */
        for (k = 1; k < WARP_SIZE; k *= 2) {
            if (tx % (2*k) == 0) {
                vals[ty][tx] += vals[ty][tx+k];
            }
            //__syncthreads();
        }
        // End of two attempts
        y[row_in_grid] = vals[threadIdx.y][0];
    }
}

/**
 * Cuda kernel for: ELL(A)x = y
 */
__global__ void k_ell_mat_vec_mm ( int N, int num_cols_per_row , int * indices,
        float * data , float * x , float * y ) {
    //TODO, but not this time: ellpack kernel
}

/**
 * Perform: CSR(A)x = y
 */
void CSRmatvecmult(int* ptr, int* J, float* Val, int N, int nnz, float* x, float *y, bool bVectorized) {
    int *ptr_d, *J_d;
    float *Val_d, *x_d, *y_d;

    /************************/
    /* copy to device       */
    /************************/

    cudaMalloc((void **) &ptr_d, (N+1) * sizeof(int));
    checkCUDAError();
    cudaMemcpy(ptr_d, ptr, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &J_d, nnz * sizeof(int));
    checkCUDAError();
    cudaMemcpy(J_d, J, nnz * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &Val_d, nnz * sizeof(float));
    checkCUDAError();
    cudaMemcpy(Val_d, Val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &x_d, N * sizeof(float));
    checkCUDAError();
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &y_d, N * sizeof(float));
    checkCUDAError();
    cudaMemcpy(y_d, y, N * sizeof(float) , cudaMemcpyHostToDevice);
    checkCUDAError();

    /************************/
    /* start kernel         */
    /************************/

    if (bVectorized) {
        //TODO: define grid and block size correctly
        dim3 grid(1, (N - 1)/WARP_SIZE + 1, 1);
        dim3 block(WARP_SIZE, WARP_SIZE, 1);

        k_csr2_mat_vec_mm<<<grid, block>>>(ptr_d, J_d, Val_d, N, x_d, y_d);
    } else {
        dim3 grid((N - 1)/TILE_SIZE + 1, 1, 1);
        dim3 block(TILE_SIZE, 1, 1);

        k_csr_mat_vec_mm<<<grid, block>>>(ptr_d, J_d, Val_d, N, x_d, y_d);
    }

    checkCUDAError();

    /************************/
    /* copy back            */
    /************************/

    cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError();

    /************************/
    /* free memory          */
    /************************/
    cudaFree(ptr_d);
    cudaFree(J_d);
    cudaFree(Val_d);
    cudaFree(x_d);
    cudaFree(y_d);
}

/**
 * Perform: ELL(A)x = y
 */
void ELLmatvecmult(int N, int num_cols_per_row , int * indices,
        float * data , float * x , float * y) {
    int *indices_d;
    float *data_d, *x_d, *y_d;

    /************************/
    /* copy to device       */
    /************************/

    cudaMalloc((void **) &indices_d, N * num_cols_per_row * sizeof(int));
    checkCUDAError();
    cudaMemcpy(indices_d, indices, N * num_cols_per_row * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &data_d, N * num_cols_per_row * sizeof(float));
    checkCUDAError();
    cudaMemcpy(data_d, data, N * num_cols_per_row * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &x_d, N * sizeof(float));
    checkCUDAError();
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError();

    cudaMalloc((void **) &y_d, N * sizeof(float));
    checkCUDAError();
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError();

    /************************/
    /* start kernel         */
    /************************/

    //NYI: define grid and block size
    //k_ell_mat_vec_mm <<< grid, block >>> (N, num_cols_per_row, indices_d, data_d , x_d, y_d);
    checkCUDAError();

    /************************/
    /* copy back            */
    /************************/

    cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError();

    /************************/
    /* free memory          */
    /************************/

    cudaFree(indices_d);
    cudaFree(data_d);
    cudaFree(x_d);
    cudaFree(y_d);
}

