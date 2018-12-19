/*
 * main.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru, meister
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"
#include "sparseformats.h"

typedef int bool;

extern void pageRank(int* ptr, int* J, float* Val, const int N, const int nnz, const bool bVectorizedCSR);
extern void poisson(const int N, const bool bCuSparse);

int M, N, nnz;
int i, *I, *J, *ptr;
float *Val;

/**
 * Read Matrix from disk.
 */
void loadMarketMatrix(char *file) {
	int ret_code;
	MM_typecode matcode;
	FILE *f;

	if ((f = fopen(file, "r")) == NULL) {
		printf("Could not open matrix file %s.\n", file);
		exit(1);
	}

	if (mm_read_banner(f, &matcode) != 0) {
	        printf("Could not process Matrix Market banner.\n");
	        exit(1);
	    }

	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */

	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
			mm_is_sparse(matcode) ) {
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	/* find out size of sparse matrix .... */

	ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz);
	if (ret_code != 0) {
		exit(1);
	}

	if (M != N) {
		printf("Sorry, matrix is not square.");
		exit(1);
	}

	if (mm_is_symmetric(matcode)) {
		nnz *= 2;

		/* reserve memory for host matrices */

		I = (int *) malloc(nnz * sizeof(int));
		J = (int *) malloc(nnz * sizeof(int));
		Val = (float *) malloc(nnz * sizeof(float));
		ptr = (int *) malloc((N+1) * sizeof(int));

		// read non-zero matrix entries
		for (i=0; i<nnz; i+=2)
		{
			//transpose matrix while reading
			fscanf(f, "%d %d\n", &J[i], &I[i]);
			--I[i];  /* adjust from 1-based to 0-based */
			--J[i];
			I[i+1] = J[i];
			J[i+1] = I[i];
			Val[i] = 1;
			Val[i+1] = 1;
		}
	} else {
		/* reserve memory for host matrices */

		I = (int *) malloc(nnz * sizeof(int));
		J = (int *) malloc(nnz * sizeof(int));
		Val = (float *) malloc(nnz * sizeof(float));
		ptr = (int *) malloc((N+1) * sizeof(int));

		// read non-zero matrix entries
		for (i=0; i<nnz; i++)
		{
			//transpose matrix while reading
			fscanf(f, "%d %d\n", &J[i], &I[i]);
			--I[i];  /* adjust from 1-based to 0-based */
			--J[i];
			Val[i] = 1;
		}
	}


	if (f !=stdin) fclose(f);

	// convert to CSR
	COOToCSR(I, J, Val, ptr, N, nnz);

	// write out matrix information
	mm_write_banner(stdout, matcode);
	mm_write_mtx_crd_size(stdout, N, N, nnz);
}

int main(int argc, char *argv[])
{
	bool bPageRank = 0, bHeatEquation = 0, bCuSparse = 0;
	bool bVectorizedCSR = 0;
	int heat_eq_size = 64;
	int i = 1;

	if (argc <= i) {
		fprintf(stderr, "Usage: %s [-P [-V] <market_matrix_filename>] [-H [-C] <matrix_size>]\n", argv[0]);
		exit(1);
	}

	for (i = 1; i < argc;) {
		if (strcmp(argv[i], "-P") == 0) {
			++i;

			if (i < argc && strcmp(argv[i], "-V") == 0) {
				i++;
				bVectorizedCSR = 1;
				printf("PageRank with CSR (vectorized)\n");
			} else {
				printf("PageRank with CSR (scalar)\n");
			}

			if (argc <= i) {
				fprintf(stderr, "PageRank requires market-matrix-filename\n");
				exit(1);
			}

			/************************/
			/* read market matrix   */
			/************************/

			loadMarketMatrix(argv[i++]);

			/************************/
			/* run page rank        */
			/************************/

			pageRank(ptr, J, Val, N, nnz, bVectorizedCSR);
		} else if (strcmp(argv[i], "-H") == 0) {
			++i;

			if (i < argc && strcmp(argv[i], "-C") == 0) {
				i++;
				bCuSparse = 1;
				printf("Heat equation (CuSparse)\n");
			} else {
				printf("Heat equation (Ellpack)\n");
			}

			if (argc <= i) {
				printf("Choosing default matrix size: %i\n", heat_eq_size);
			} else {
				sscanf(argv[i++], "%i", &heat_eq_size);
				printf("Matrix size: %i\n", heat_eq_size);			
			}

			/************************/
			/* solve poisson 1D     */
			/************************/

		   	poisson(heat_eq_size, bCuSparse);
		} else {
			fprintf(stderr, "Usage: %s [-P [-V] <market_matrix_filename>] [-H [-C] [<matrix_size>]]\n", argv[0]);
			exit(1);
		}
	}

    return 0;
}

