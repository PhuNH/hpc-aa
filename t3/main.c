/*
 * main.c
 *
 *  Created on: Jan 13, 2012
 *      Author: butnaru, meister
 */

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "sparseformats.h"

typedef int bool;

extern void pageRank(int* ptr, int* J, float* Val, int N, int nnz, bool bVectorizedCSR);
extern void poisson(int N);

int M, N, nnz;
int i, *I, *J, *ptr;
float *Val;

/**
 * Read Matrix from disk.
 */
loadMarketMatrix(char *file) {
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
			I[i]--;  /* adjust from 1-based to 0-based */
			J[i]--;
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
			I[i]--;  /* adjust from 1-based to 0-based */
			J[i]--;
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
	bool bVectorizedCSR = 0;

    if (argc < 2) {
		fprintf(stderr, "Usage: %s [-V] <market-matrix-filename>\n", argv[0]);
		exit(1);
	}

	if (argc >= 2 && strcmp(argv[1], "-V") == 0) {
		bVectorizedCSR = 1;
		printf("PageRank algorithm with CSR (vectorized)\n");
	} else {
		printf("PageRank algorithm with CSR (scalar)\n");
	}

	/************************/
	/* read market matrix   */
	/************************/
    loadMarketMatrix(argv[argc - 1]);

	/************************/
	/* run page rank        */
	/************************/

    pageRank(ptr, J, Val, N, nnz, bVectorizedCSR);

	/************************/
	/* solve poisson 1D     */
	/************************/

   // poisson(atoi(argv[1]));

	return 0;
}

