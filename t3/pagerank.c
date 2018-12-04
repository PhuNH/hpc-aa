/*
 * pagerank.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru, meister
 */

#include <stdlib.h>
#include <stdio.h>
#include "math.h"

typedef int bool;

void pageRank(int* ptr, int* J, float* Val, int N, int nnz, bool bVectorizedCSR) {
	const float alpha = 0.85;
	const float EPS = 1.0e-4;
	float *x = (float *) malloc(N * sizeof(float));
	float *y = (float *) malloc(N * sizeof(float));
	int *ni = (int *) calloc(N, sizeof(float));
	float err, sum, newX;
	int i, j;

	/************************/
	/* prepare B            */
	/************************/

	// count outgoing links. matrix entry i, j describes
	// incoming links to i from j, so we need a column sum
	for (i = 0; i < nnz; ++i) {
		ni[J[i]] += 1;
	}

	for (i = 0; i < N; ++i) {
		if (ni[i] == 0) {
			printf("\rWarning: Column %i sum is zero, non-stochastic matrix!", i);
            break;
		}
	}

	printf("\n");

	// weight columns by nr of outgoing links (previously calculated)
	for (i = 0; i < nnz; ++i) {
		Val[i] /= ni[J[i]];
	}

	/************************/
	/* prepare x            */
	/************************/

	// init x vector with 1/N, y with 0
	for (i = 0; i < N; ++i) {
		x[i] = 1.0/N;
		y[i] = 0;
	}

	/************************/
	/* find eigenvalue      */
	/************************/
	err = 2.0 * EPS; //choose something bigger than EPS initially
	
	for (i = 1; err > EPS; i++) {
		// compute y += Bx
		CSRmatvecmult(ptr, J, Val, N, nnz, x, y, bVectorizedCSR);

		err = 0.0;
		sum = 0.0;

		for (j = 0; j < N; ++j) {
			// do regularization
			newX = alpha * y[j] + (1.0 - alpha) * 1.0/N;

			// calculate error
			err += fabs(x[j] - newX);

			// replace old x with new x
			x[j] = newX;
			y[j] = 0;
			sum += x[j];
		}

		printf("Iterations = %i, err = %e, sum = %f\n", i, err, sum);
	}

	printf("\n\nSolution: \n");

	/************************/
	/* print solution       */
	/************************/

    int i_max = 0;
    float x_max = 0.0f;

	for (i = 0; i < N; ++i) {
        if (x[i] > x_max) {
            i_max = i;
            x_max = x[i];
        }

        if (i < 10) {
		    printf("x_%d = %e\n", i + 1, x[i]);
        }
	}

    printf("\nMaximum:\nx_%d = %e\n", i_max + 1, x_max);
}
