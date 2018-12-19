#ifndef __KERNELS_H_
#define __KERNELS_H_

void CSR_matvec(int N, int nnz, int* start, int* indices, float* data, float* x, float *y, int bVectorized);
void CSR_create(int N, int nnz, int* start, int * indices, float * data , float * x , float * y, int** start_d, int **indices_d, float **data_d, float **x_d, float **y_d);
void CSR_kernel(int N, int nnz, int* start_d, int * indices_d, float * data_d , float * x_d , float * y_d, int bVectorized);
void CSR_destroy(int* start_d, int* indices_d, float* data_d, float* x_d, float* y_d);

void ELL_matvec(int N, int num_cols_per_row , int * indices, float * data , float * x , float * y);
void ELL_create(int N, int num_cols_per_row, int * indices, float * data , float * x , float * y, int **indices_d, float **data_d, float **x_d, float **y_d);
void ELL_kernel(int N, int num_cols_per_row , int * indices_d, float * data_d , float * x_d , float * y_d);
void ELL_destroy(int* indices_d, float* data_d, float* x_d, float* y_d);

#endif
