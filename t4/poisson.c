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
#define checkCusparseError(x)	if ((cusparseStatus = (cublasStatus_t) (x))) { printf("Cusparse Error %i: %s(%i)\n%s\n", cusparseStatus, __FILE__, __LINE__, #x); exit(1);}

typedef int bool;

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
	for (i = 0; i < N-1; i++) {
		indices[N * 0 + i] = i-1;   data[N * 0 + i] = 1;
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
		// TODO: err = || y_d || 
		// TODO: x_d = x_d + dt / (dx * dx) * c * y_d
		if ((i & 511) == 0 || err > epsilon) {
			// Copy back for output.
			// TODO: x = x_d
            #if defined GNUPLOT
    		    static struct timespec sleep_time;
    		    sleep_time.tv_sec = 0;
    		    sleep_time.tv_nsec = 40000000;
    		    static struct timespec remaining;
    		    gnuplot_resetplot(h_gnuplot);
			    sprintf(s_tmp, "Temperature (t = %.4f, err = %.4e)", dt * i, err);
			    gnuplot_plot_xf(h_gnuplot, x, N, s_tmp);
			    nanosleep(&sleep_time, &remaining);
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
	// TODO: Homework!
}

void poisson(const int N, const bool b_CuSparse) {
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

	if (b_CuSparse) {
		poisson_cusparse(N, c, epsilon, x, y);
	} else {
		poisson_ellpack(N, c, epsilon, x, y);
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
