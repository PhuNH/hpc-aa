/*
 * pagerank.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru, meister
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "gnuplot_i.h"
#include "kernels.h"
typedef int bool;

void pageRank(int* ptr, int* J, float* Val, const int N, const int nnz, const bool bVectorizedCSR) {
	const float alpha = 0.85, epsilon = 1.0e-4;
	float *x = (float *) malloc(N * sizeof(float));
	float *y = (float *) malloc(N * sizeof(float));
	int *ni = (int *) calloc(N, sizeof(float));
	float err, sum, newX;
	int i, j;
	char s_tmp[64];
	gnuplot_ctrl* h_gnuplot;

	h_gnuplot = gnuplot_init();
	sprintf(s_tmp, "set xrange [0:%i]", N - 1);
	gnuplot_cmd(h_gnuplot, s_tmp);
	gnuplot_setstyle(h_gnuplot, "boxes");
	gnuplot_set_xlabel(h_gnuplot, "Page");
	gnuplot_set_ylabel(h_gnuplot, "Rank");

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
	err = 2.0f * epsilon; //choose something bigger than epsilon initially
	
	for (i = 1; err > epsilon; i++) {
		// compute y = B x
		CSR_matvec(N, nnz, ptr, J, Val, x, y, bVectorizedCSR);

		err = 0.0f;
		sum = 0.0f;

		for (j = 0; j < N; ++j) {
			// do regularization x' = a y + (1 - a) / N * e
			newX = alpha * y[j] + (1.0f - alpha) * 1.0f/N;

			// calculate error
			err += fabs(x[j] - newX);

			// replace x with x'
			x[j] = newX;
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

    printf("\nMaximum:\nx_%d = %e\n", i_max, x_max);

	gnuplot_plot_xf(h_gnuplot, x, N, "Rank");

	printf("Press enter to continue");
	getc(stdin);
	gnuplot_close(h_gnuplot);
}
