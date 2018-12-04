#include "stdlib.h"

/**
 * A single matrix element as struct: [i j val].
 */
typedef struct {
	int i;
	int j;
	float val;
} mat_elem;

/**
 * Sorting function for two mat_elem
 */
int cmp_by_ij(const void *a, const void *b)
{
    mat_elem *ia = (mat_elem*) a;
    mat_elem *ib = (mat_elem*) b;

    if (ia->i > ib->i) {
    	return 1;
    } else if (ia->i < ib->i) {
    	return -1;
    } else {
    	if (ia->j > ib->j) {
			return 1;
		} else if (ia->j < ib->j) {
			return -1;
		} else {
			return 0;
		}
    }
}

/**
 * Converts from COO to CSR.
 */
void COOToCSR(int* I, int* J, float* Val, int* ptr, int N, int nnz) {
	mat_elem *m = (mat_elem*) calloc(nnz, sizeof(mat_elem));
	int x, currRow;

	// use additional structure for sorting
	for (x = 0; x < nnz; ++x) {
		m[x].i = I[x];
		m[x].j = J[x];
		m[x].val = Val[x];
	}

	// sort after collumn and then after i
	qsort(m, nnz, sizeof(mat_elem), cmp_by_ij);

	// set all ptr to 0
	for (x = 0; x < N+1; ++x) {
		ptr[x] = -1;
	}

	// fill in ptr
	currRow = 0;
	ptr[currRow] = 0;
	for (x = 0; x < nnz; ++x) {
		if (m[x].i > currRow) {
			currRow = m[x].i;
			ptr[currRow] = x;
		}

		J[x] = m[x].j;
		Val[x] = m[x].val;
	}

	// last element in ptr is set to nnz
	ptr[N] = nnz;

	// all missing rows get the ptr of the previous row
	for (x = N-1; x >= 0; x--) {
		if (ptr[x] == -1) {
			ptr[x] = ptr[x+1];
		}
	}
}
