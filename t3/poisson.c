/*
 * poisson.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru
 */

#include <stdlib.h>
#include <stdio.h>

void poisson(int N) {
	int i;
	int num_cols_per_row = 3;
	int* indices = (int*) calloc(N * num_cols_per_row, sizeof(int));
	float* data  = (float*) calloc(N* num_cols_per_row, sizeof(float));
	float* x  = (float*) calloc(N, sizeof(float));
	float* y  = (float*) calloc(N, sizeof(float));

	// fill matrix with stencil [-1 2 -1]
	for (i = 1; i < N-1; i ++) {
		data[i*num_cols_per_row] = -1;
		indices[i*num_cols_per_row] = i-1;

		data[i*num_cols_per_row +1] = 2;
		indices[i*num_cols_per_row +1] = i;

		data[i*num_cols_per_row + 2] = -1;
		indices[i*num_cols_per_row + 2] = i+1;
	}

	// first and last line
	data[0] = 2;
	indices[0] = 0;
	data[1] = -1;
	indices[1] = 1;
	data[(N-1)*num_cols_per_row+1] = -1;
	indices[(N-1)*num_cols_per_row+1] = N-2;
	data[(N-1)*num_cols_per_row+2] = 2;
	indices[(N-1)*num_cols_per_row+2] = N-1;





	// set initial state x = 1 (except for boundaries x[0] and x[N-1] both = 0)
	for (i = 0; i < N; ++i) {
		// main diagonal
		x[i] = 1;
	}

	ELLmatvecmult(N, num_cols_per_row , indices, data , x , y);

	for (i = 0; i < N*num_cols_per_row; ++i) {
		printf("indices[%d] = %d\n", i, indices[i]);
	}

	for (i = 0; i < N*num_cols_per_row; ++i) {
		printf("data[%d] = %f\n", i, data[i]);
	}


	for (i = 0; i < N; ++i) {
		printf("x[%d] = %f\n", i, x[i]);
	}

	for (i = 0; i < N; ++i) {
		printf("y[%d] = %f\n", i, y[i]);
	}
}
