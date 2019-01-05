/*
 * poisson.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

//#define GNUPLOT

#if defined GNUPLOT
#   include "gnuplot_i.h"
    gnuplot_ctrl* h_gnuplot;
#endif

#include "kernels.h"

#define checkCublasError(x)		if ((cublasStatus = (cublasStatus_t) (x))) { printf("Cublas Error %i: %s(%i)\n%s\n", cublasStatus, __FILE__, __LINE__, #x); exit(1);}
#define checkCusparseError(x)	if ((cusparseStatus = (x))) { printf("Cusparse Error %i: %s(%i)\n%s\n", cusparseStatus, __FILE__, __LINE__, #x); exit(1);}

enum Kernel {
    ELLPACK, Band, cuSPARSE
};

typedef int bool;

void poisson_band(const int N, const float c, const float epsilon, float* x, float* y) {
	const int k_max = 1;
	float* data_d, *data  = (float*) calloc(N*(2 * k_max + 1), sizeof(float));
	float* x_d, *y_d;
	float alpha, err;
	int i, j;
	float dx = 1.0 / (float)(N - 1);
	float dt = (0.5f * dx * dx) / c;
	char s_tmp[64];

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    checkCublasError(cublasCreate(&cublasHandle));

	// fill matrix with stencil [1 -2 1]
	for (i = 0; i < N; i++) {
		data[i + N * 0] = 1;
		data[i + N * 1] = -2;
		data[i + N * 2] = 1;
	}

	// first and last line (Dirichlet condition)

	data[0 + N * 0] = 0;
	data[0 + N * 1] = -2;
	data[(N - 1) + N * 1] = -2;
	data[(N - 1) + N * 2] = 0;

	band_create(N, 2 * k_max + 1, data, x, y, &data_d, &x_d, &y_d);

	err = 2.0f * epsilon; //choose something bigger than epsilon initially

	for (i = 0; err > epsilon; ++i) {
		band_kernel(N, k_max, data_d, x_d, y_d);
	
    	checkCublasError(cublasSnrm2(cublasHandle, N, y_d, 1, &err));

		alpha = dt/(dx * dx) * c;
	    checkCublasError(cublasSaxpy(cublasHandle, N, &alpha, y_d, 1, x_d, 1));

		if ((i & 511) == 0) {
			checkCublasError(cudaMemcpy(x, x_d, N * sizeof(float), cudaMemcpyDeviceToHost));

            #if defined GNUPLOT
    		    gnuplot_resetplot(h_gnuplot);
			    sprintf(s_tmp, "Temperature (t = %.4f, err = %.4e)", dt * i, err);
			    gnuplot_plot_xf(h_gnuplot, x, N, s_tmp);
            #endif

            printf("t = %.4f, err = %.4e, Temperature at x = 0.5: %.4e\n", dt * i, err, x[N/2]);
		}
	}

	band_destroy(data_d, x_d, y_d);

    cublasDestroy(cublasHandle);

	free(data);
}

void poisson_ellpack(const int N, const float c, const float epsilon, float* x, float* y) {
	const int num_cols_per_row = 3;
	int* indices_d, *indices = (int*) calloc(N * num_cols_per_row, sizeof(int));
	float* data_d, *data  = (float*) calloc(N* num_cols_per_row, sizeof(float));
	float* x_d, *y_d;
	float alpha, err;
	int i, j;
	float dx = 1.0 / (float)(N - 1);
	float dt = (0.5f * dx * dx) / c;
	char s_tmp[64];

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    checkCublasError(cublasCreate(&cublasHandle));

	// fill matrix with stencil [1 -2 1]
	for (i = 1; i < N-1; i++) {
		indices[i] = i-1;           data[i] = 1;
		indices[N * 1 + i] = i;     data[N * 1 + i] = -2;
		indices[N * 2 + i] = i+1;   data[N * 2 + i] = 1;
	}

	// first and last line (Dirichlet condition)
	
	indices[N * 1 + 0] = 0;         data[N * 1 + 0] = -2;
	indices[N * 2 + 0] = 1;         data[N * 2 + 0] = 1;

	indices	[N * 0 + N - 1] = N-2;  data[N * 0 + N - 1] = 1;
	indices	[N * 1 + N - 1] = N-1;  data[N * 1 + N - 1] = -2;

	ELL_create(N, num_cols_per_row, indices, data, x, y, &indices_d, &data_d, &x_d, &y_d);

	err = 2.0f * epsilon; //choose something bigger than epsilon initially

	for (i = 0; err > epsilon; ++i) {
		ELL_kernel(N, num_cols_per_row, indices_d, data_d, x_d, y_d);
    	checkCublasError(cublasSnrm2(cublasHandle, N, y_d, 1, &err));

		alpha = dt/(dx * dx) * c;
	    checkCublasError(cublasSaxpy(cublasHandle, N, &alpha, y_d, 1, x_d, 1));

		if ((i & 511) == 0) {
			checkCublasError(cudaMemcpy(x, x_d, N * sizeof(float), cudaMemcpyDeviceToHost));

            #if defined GNUPLOT
    		    gnuplot_resetplot(h_gnuplot);
			    sprintf(s_tmp, "Temperature (t = %.4f, err = %.4e)", dt * i, err);
			    gnuplot_plot_xf(h_gnuplot, x, N, s_tmp);
            #endif

            printf("t = %.4f, err = %.4e, Temperature at x = 0.5: %.4e\n", dt * i, err, x[N/2]);
		}
	}

	ELL_destroy(indices_d, data_d, x_d, y_d);

    cublasDestroy(cublasHandle);

	free(indices);
	free(data);
}

void poisson_cusparse(const int N, const float c, const float epsilon, float* x, float* y) {
	const int num_cols_per_row = 3;
	int* start_d, *start = (int*) calloc(N + 1, sizeof(int));
	int* indices_d, *indices = (int*) calloc(N * num_cols_per_row, sizeof(int));
	float* data_d, *data  = (float*) calloc(N* num_cols_per_row, sizeof(float));
	float* x_d, *y_d;
	float err, alpha, beta;
	int i, j;
	float dx = 1.0 / (float)(N - 1);
	float dt = (0.5f * dx * dx) / c;
	char s_tmp[64];

	for (i = 0; i <= N; i++) {
		start[i] = 3 * i;
	}	

	// fill matrix with stencil [1 -2 1]
	for (i = 1; i < N-1; i++) {
		
		indices[num_cols_per_row * i] = i-1;        data[num_cols_per_row * i] = 1;
		indices[num_cols_per_row * i + 1] = i;      data[num_cols_per_row * i + 1] = -2;
		indices[num_cols_per_row * i + 2] = i+1;    data[num_cols_per_row * i + 2] = 1;
	}

	// first and last line (Outflow condition)
	indices[num_cols_per_row * 0 + 1] = 0;          data[num_cols_per_row * 0 + 1] = -1;
	indices[num_cols_per_row * 0 + 2] = 1;          data[num_cols_per_row * 0 + 2] = 1;

	indices[num_cols_per_row * (N - 1)] = N-2;      data[num_cols_per_row * (N - 1)] = 1;
	indices[num_cols_per_row * (N - 1) + 1] = N-1;  data[num_cols_per_row * (N - 1) + 1] = -1;

	// first and last line (Dirichlet condition)
	indices[num_cols_per_row * 0 + 1] = 0;          data[num_cols_per_row * 0 + 1] = -2;
	indices[num_cols_per_row * 0 + 2] = 1;          data[num_cols_per_row * 0 + 2] = 1;

	indices[num_cols_per_row * (N - 1)] = N-2;      data[num_cols_per_row * (N - 1)] = 1;
	indices[num_cols_per_row * (N - 1) + 1] = N-1;  data[num_cols_per_row * (N - 1) + 1] = -2;

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    checkCublasError(cublasCreate(&cublasHandle));

    //TODO: Get handle for cuSPARSE
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    checkCusparseError(cusparseCreate(&cusparseHandle));
    //TODO: convert CSR matrix to cuSPARSE hybrid matrix
    cusparseHybMat_t hyb_d = NULL;
    cusparseMatDescr_t descrA = NULL;
    checkCusparseError(cusparseCreateHybMat(&hyb_d));
    checkCusparseError(cusparseCreateMatDescr(&descrA));
    checkCusparseError(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCusparseError(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    //checkCusparseError(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    //checkCusparseError(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));

    CSR_create(N, N * num_cols_per_row, start, indices, data, x , y, &start_d, &indices_d, &data_d, &x_d, &y_d);
    checkCusparseError(cusparseScsr2hyb(cusparseHandle, N, N, descrA, data_d, start_d, indices_d, hyb_d, 3, CUSPARSE_HYB_PARTITION_MAX));

	cudaFree(start_d);
	cudaFree(indices_d);
	cudaFree(data_d);

	err = 2.0f * epsilon; //choose something bigger than epsilon initially

	for (i = 0; err > epsilon; ++i) {
		alpha = 1.0f;
		beta = 0.0f;
		//TODO: Add cuSPARSE instructions
        //checkCusparseError(cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N*3, &alpha, descrA, data_d, start_d, indices_d, x_d, &beta, y_d));
        checkCusparseError(cusparseShybmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, hyb_d, x_d, &beta, y_d));

    	checkCublasError(cublasSnrm2(cublasHandle, N, y_d, 1, &err));

		alpha = dt/(dx * dx) * c;
	    checkCublasError(cublasSaxpy(cublasHandle, N, &alpha, y_d, 1, x_d, 1));

		if ((i & 511) == 0 || err <= epsilon) {
			checkCublasError(cudaMemcpy(x, x_d, N*sizeof(float), cudaMemcpyDeviceToHost));

            #if defined GNUPLOT
    		    gnuplot_resetplot(h_gnuplot);
			    sprintf(s_tmp, "Temperature (t = %.4f, err = %.4e)", dt * i, err);
			    gnuplot_plot_xf(h_gnuplot, x, N, s_tmp);
            #endif

            printf("t = %.4f, err = %.4e, Temperature at x = 0.5: %.4e\n", dt * i, err, x[N/2]);
		}
	}

	cudaFree(x_d);
	cudaFree(y_d);

	//TODO: Destroy Hybrid matrix and the cuSparse handle.
    checkCusparseError(cusparseDestroyMatDescr(descrA));
    checkCusparseError(cusparseDestroyHybMat(hyb_d));
    checkCusparseError(cusparseDestroy(cusparseHandle));

	free(start);
	free(indices);
	free(data);
}

void poisson(const int N, const enum Kernel kernel) {
	const float c = 0.1, epsilon = 1.0e-6;
	float* x  = (float*) calloc(N, sizeof(float));
	float* y  = (float*) calloc(N, sizeof(float));
	int i, j;
	char s_tmp[64];

    #if defined GNUPLOT
	    h_gnuplot = gnuplot_init();
	    sprintf(s_tmp, "set xrange [0:%i]", N - 1);
	    gnuplot_cmd(h_gnuplot, s_tmp);
	    gnuplot_cmd(h_gnuplot, "set yrange [0:1]");
	    gnuplot_setstyle(h_gnuplot, "lines");
	    gnuplot_set_xlabel(h_gnuplot, "x");
	    gnuplot_set_ylabel(h_gnuplot, "Temperature");
    #endif

	// set initial state
	for (i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 0.0f;
	}

	float t_start = clock();

    switch(kernel) {
	    case cuSPARSE:
		    poisson_cusparse(N, c, epsilon, x, y);
            break;

	    case ELLPACK:
		    poisson_ellpack(N, c, epsilon, x, y);
            break;

	    case Band:
		    poisson_band(N, c, epsilon, x, y);
            break;
    }

	float t_end = clock();

	printf("Total time: %.3f s\n", (float)(t_end - t_start) / (float)CLOCKS_PER_SEC);

    #if defined GNUPLOT
	    printf("Press enter to continue");
	    getc(stdin);
	    gnuplot_close(h_gnuplot);
    #endif

	free(x);
	free(y);
}
