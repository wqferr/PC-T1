#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <omp.h>
#include <mpi.h>

#define mat_get(m, w, i, j) (m[w*i + j])

void read_matrix(double **m, int *w, int *h);
int row_process(int row_idx, int world_size, int *counts, int w);
void print_row(double *r, int w);

enum {
	TAG_PING,
	TAG_SWAP_ROWS
};

typedef struct {
	double value;
	int row;
} column_element;

int main(int argc, char *argv[]) {
	int global_rank, world_size;
	int i, j;
	int subm_start; // first row the process is responsible for
	int subm_n_rows; // number of rows of the process's submatrix
	int curr_column_idx = 0; // current row being eliminated
	int w, h; // matrix dimensions
	double max_abs_val; // maximum absolute valued element
						// in a column of the submatrix
	int *displs = NULL; // displacements for MPI_Scatterv
	int *counts = NULL; // number of rows each process is responsible for
	int *send_counts = NULL; // counts for MPI_Scatterv
	double *m = NULL; // the entire matrix
	double *subm = NULL; // rows this process is responsible for
	double *row_aux = NULL; // row to be used for elimination
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	displs = calloc(world_size, sizeof(*displs));
	counts = malloc(world_size * sizeof(*counts));
	send_counts = malloc(world_size * sizeof(*send_counts));

	// ============= Init matrix ============
	if (global_rank == 0) {
		int rows_per_process;	// min number of rows per process
								// remaining rows are distributed to processes
								// with lowest ranks

		read_matrix(&m, &w, &h);

		rows_per_process = h / world_size;

		// set min rows per process
		for (i = 0; i < world_size; i++) {
			counts[i] = rows_per_process;
			send_counts[i] = w*rows_per_process;
		}
		// add single row to process so as to distribute load
		for (i = 0; i < h % world_size; i++) { // distribute remaining rows
			counts[i]++;
			send_counts[i] += w;
		}

		// set arrays for MPI_Scatterv
		for (i = 1; i < world_size; i++) {
			displs[i] = displs[i-1] + counts[i-1]*w;
		}
	}

	// ========== End init ==========

	// ========== Broadcast information to other processes =========

	// broadcast matrix width
	// height is only necessary for the root process
	MPI_Bcast(
		&w,
		1, MPI_INT,
		0, // root
		MPI_COMM_WORLD);

	MPI_Bcast(
		counts, 		// buffer
		world_size, MPI_INT, // block description
		0,				// root
		MPI_COMM_WORLD);

	// find the index of the submatrix's first row
	subm_start = 0;
	for (i = 0; i < global_rank; i++) {
		// sum of the length of all the previous submatrices
		subm_start += counts[i];
	}

	// allocate submatrix given number of elements
	subm = malloc(counts[global_rank]*w * sizeof(*subm));
	subm_n_rows = counts[global_rank];

	MPI_Scatterv(
		m,				// send buffer
		send_counts, displs,	// send block size and displacement
		MPI_DOUBLE,		// send type
		subm,			// receive buffer
		counts[global_rank]*w,	// receive count
		MPI_DOUBLE,		// receive type
		0,				// root
		MPI_COMM_WORLD);
	free(send_counts);
	free(displs);

	// ======== Begin processing =======
	row_aux = malloc(w * sizeof(*row_aux));

	// for every column in the matrix
	// the iterations are synchronized across all processes
	for (curr_column_idx = 0; curr_column_idx < w; curr_column_idx++) {
		column_element best_local, best;

		best_local.value = -1;
		best.value = -1;

		if (global_rank == 0) {
			printf("========\n\nEliminating column %d\n", curr_column_idx);
		}
		// DEBUG BARRIER
		MPI_Barrier(MPI_COMM_WORLD);

		// if this row was not already processed
		if (global_rank >= curr_column_idx) {
			// then it could potentially be swapped with the current row

			// find local best value in the submatrix column
			#pragma omp parallel for
			for (i = 0; i < subm_n_rows; i++) {
				double v = abs(subm[i*w + curr_column_idx]);
				//printf("subm[%d*%d + %d] = %lf @ p%d\n", i, w, curr_column_idx, v, global_rank);
				if (v > best_local.value) {
					#pragma omp critical
					{
						best_local.value = v;
						best_local.row = subm_start + i;
					}
				}
			}
		}

		printf("best local: %lf %d @ p%d\n", best_local.value, best_local.row, global_rank);

		// find the global best among all local bests
		// if the row wasn't eligible for swapping, best_local is -1 and thus
		// will never be selected
		MPI_Allreduce(
			&best_local,	// send buffer
			&best,			// recv buffer
			1,				// send count
			MPI_DOUBLE_INT,	// datatype
			MPI_MAXLOC,		// operation
			MPI_COMM_WORLD);

		printf("best: %lf %d @ p%d\n", best.value, best.row, global_rank);
		// TODO check for 0

		printf("best.row=%d\tcurr_column_idx=%d\n", best.row, curr_column_idx);
		// ======= Row swap ======
		if (best.row != curr_column_idx) { // must swap rows
			int proc_best = row_process(best.row, world_size, counts, w);
			int proc_curr = row_process(curr_column_idx, world_size, counts, w);

			if (global_rank == 0) {
				printf("best proc: %d\tcurr proc: %d\n", proc_best, proc_curr);
			}
			if (proc_best == proc_curr) {
				if (proc_best == global_rank) {
					int row1_idx = curr_column_idx - subm_start;
					int row2_idx = best.row - subm_start;

					double *row1 = &subm[row1_idx*w];
					double *row2 = &subm[row2_idx*w];

					// swap rows
					memcpy(row_aux, row1, w*sizeof(*subm));
					memcpy(row1, row2, w*sizeof(*subm));
					memcpy(row2, row_aux, w*sizeof(*subm));
				}
			} else {
				if (proc_best == global_rank) { // process contains best row
					int row_idx = best.row - subm_start;

					MPI_Send(
						&subm[row_idx*w],	// send buffer
						w, MPI_DOUBLE,		// type and number
						proc_curr,			// dest
						TAG_SWAP_ROWS,		// tag
						MPI_COMM_WORLD);
					MPI_Recv(
						row_aux,
						w, MPI_DOUBLE,
						proc_curr,
						TAG_SWAP_ROWS,
						MPI_COMM_WORLD,
						&status);
					memcpy(&subm[row_idx*w], row_aux, w*sizeof(*subm));
				} else if (proc_curr == global_rank) {
					int row_idx = curr_column_idx - subm_start;

					MPI_Send(
						&subm[row_idx*w],	// send buffer
						w, MPI_DOUBLE,		// type and number
						proc_best,			// dest
						TAG_SWAP_ROWS,		// tag
						MPI_COMM_WORLD);
					MPI_Recv(
						row_aux,
						w, MPI_DOUBLE,
						proc_best,
						TAG_SWAP_ROWS,
						MPI_COMM_WORLD,
						&status);
					memcpy(&subm[row_idx*w], row_aux, w*sizeof(*subm));
				}
			}
		}

		// TODO broadcast and eliminate other entries in this column
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// gather final matrix to root

	// BEGIN DEBUG CODE
	if (global_rank != 0) {
		// wait until previous process has finished printing
		MPI_Recv(
			NULL,		// doesnt matter, empty message
			0,			// recv count
			MPI_INT,	// doesnt matter
			MPI_ANY_SOURCE,	// could be global_rank - 1
			TAG_PING,	// tag
			MPI_COMM_WORLD,
			&status);
	}

	printf("\nProcess %d received:", global_rank);
	for (i = 0; i < counts[global_rank]*w; i++) {
		if (i % w == 0) {
			printf("\n");
		}
		printf("%lf ", subm[i]);
	}
	printf("\n");

	if (global_rank + 1 < world_size) { // is not the highest ranked process
		MPI_Send( // ping next process to print their submatrix
			NULL,		// buffer
			0,			// send count
			MPI_INT,	// any datatype will do
			global_rank + 1,	// dest = self
			TAG_PING,			// tag
			MPI_COMM_WORLD);
	}
	// END DEBUG CODE
	/**/

	free(counts);
	free(row_aux);
	free(m);
	free(subm);
	MPI_Finalize();
	return 0;
}

void read_matrix(double **matrix, int *width, int *height) {
	int i, j;
	int w, h, n;
	double *m;

	scanf("%d%d", &h, &w);

	n = w*h;
	m = malloc(n * sizeof(*m));
	for (i = 0; i < n; i++) {
		scanf("%lf", &m[i]);
	}

	*matrix = m;
	*width = w;
	*height = h;
}

int row_process(int row_idx, int world_size, int *counts, int w) {
	int i;

	for (i = 0; i < world_size; i++) {
		if (row_idx < counts[i]) {
			return i;
		}

		// get the index relative to the next process
		row_idx -= counts[i];
	}

	return -1;
}

void print_row(double *r, int w) {
	int i;
	for (i = 0; i < w; i++) {
		printf("%lf ", r[i]);
	}
	printf("\n");
}