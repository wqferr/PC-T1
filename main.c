#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <omp.h>
#include <mpi.h>

#define mat_get(m, w, i, j) ((m)[(i)*(w) + (j)])
#define mat_row(m, w, i) (&((m)[(i)*(w)]))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define MATRIX_FILENAME "matriz.txt"
#define VECTOR_FILENAME "vetor.txt"
#define OUT_FILENAME "resultado.txt"

void read_matrix(double **m, int *w, int *h);
int row_find_proc(int row_idx, int world_size, int *counts);
void row_swap(
	int best_row, int curr_col, int *counts, int w,
	int world_size, double *subm, int subm_start, int global_rank,
	double *row_aux);
void row_normalize(double *row, int col, int w);
void row_elim_col(
	const double *row, double *dest_row, int w, int elim_col);

enum {
	TAG_PING,
	TAG_row_swap
};

typedef struct {
	double value;
	int row;
} column_element;

int main(int argc, char *argv[]) {
	int global_rank, world_size;
	int i;
	int subm_start; // first row the process is responsible for
	int subm_n_rows; // number of rows of the process's submatrix
	int curr_column_idx = 0; // current row being eliminated
	int w, h; // matrix dimensions
	int max_elim_col;
	double t;
	int *displs = NULL; // displacements for MPI_Scatterv
	int *counts = NULL; // number of rows each process is responsible for
	int *send_counts = NULL; // counts for MPI_Scatterv
	double *m = NULL; // the entire matrix
	double *subm = NULL; // rows this process is responsible for
	double *row_aux = NULL; // row to be used for elimination
	MPI_Status status;
	FILE *of;

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
		max_elim_col = min(w, h);

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

		t = omp_get_wtime();
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
		&max_elim_col,
		1, MPI_INT,
		0,
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

	// ======== Begin processing =======
	row_aux = malloc(w * sizeof(*row_aux));

	// for every column in the matrix
	// the iterations are synchronized across all processes
	for (curr_column_idx = 0; curr_column_idx < max_elim_col; curr_column_idx++) {
		column_element best_local, best;
		int elim_row_proc;
		if (global_rank == 0)
			printf("elim col %d\t\t @p%d\n", curr_column_idx, global_rank);

		best_local.value = -1;
		best.value = -1;

		// find local best value in the submatrix column
		#pragma omp parallel for schedule(dynamic)
		for (i = 0; i < subm_n_rows; i++) {
			if (subm_start + i >= curr_column_idx) {
				// row is candidate for swapping
	
				double v = abs(mat_get(subm, w, i, curr_column_idx));
				#pragma omp critical
				{
					if (v > best_local.value) {
						best_local.value = v;
						best_local.row = subm_start + i;
					}
				}
			}
		}
		printf("a@p%d\n", global_rank);

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

		printf("b@p%d\n", global_rank);

		// TODO row swap and elimination in a single broadcast
		// ======= Row swap ======
		if (best.row != curr_column_idx) { // must swap rows
			if (global_rank == 0)
				printf("swap %d@p%d and %d@p%d\n", best.row, row_find_proc(best.row, world_size, counts), curr_column_idx, row_find_proc(curr_column_idx, world_size, counts));
			row_swap(
				best.row, curr_column_idx, counts, w,
				world_size, subm, subm_start, global_rank, row_aux);
		}

		elim_row_proc = row_find_proc(curr_column_idx, world_size, counts);
		if (global_rank == elim_row_proc) {
			double *subm_row = mat_row(subm, w, curr_column_idx - subm_start);
			row_normalize(subm_row, curr_column_idx, w);
			memcpy(row_aux, subm_row, w*sizeof(*subm));
		}
		printf("c@p%d\n", global_rank);

		MPI_Bcast(
			row_aux,
			w, MPI_DOUBLE,
			elim_row_proc,
			MPI_COMM_WORLD);

		#pragma omp parallel for num_threads(8)
		for (i = 0; i < subm_n_rows; i++) {
			if (i + subm_start != curr_column_idx) {
				row_elim_col(row_aux, mat_row(subm, w, i), w, curr_column_idx);
			}
		}

		printf("d@p%d\n", global_rank);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gatherv(
		subm,		// send buf
		w*counts[global_rank],	// send count
		MPI_DOUBLE,	// send type
		m,			// recv buf
		send_counts,
		displs,		// recv displacements
		MPI_DOUBLE,	// recv type
		0,			// root
		MPI_COMM_WORLD);

	if (global_rank == 0) {
		t = omp_get_wtime() - t;
		printf("finished in %lf seconds\n", t);

		of = fopen(OUT_FILENAME, "w+");
		for (i = 0; i < h; i++) {
			fprintf(of, "%.3lf\n", m[(i+1)*w - 1]);
		}
		fclose(of);
	}

	free(send_counts);
	free(displs);
	free(counts);
	free(row_aux);
	free(m);
	free(subm);
	MPI_Finalize();
	return 0;
}

void read_matrix(double **matrix, int *width, int *height) {
	int i, j;
	int w, h;
	char c;
	double *m;
	FILE *mf;
	FILE *vf;

	mf = fopen(MATRIX_FILENAME, "r");
	vf = fopen(VECTOR_FILENAME, "r");

	// count number of elements in the first row
	w = 1;
	while ((c = fgetc(mf)) != '\n') {
		if (c == ' ') {
			w++;
		}
	}
	h = w; // assume matrix is square
	rewind(mf);

	// assume it is a valid matrix and all rows have the same length

	w++; // account for extra column for the vector

	m = malloc(w*h * sizeof(*m));
	for (i = 0; i < h; i++) {
		for (j = 0; j < (w-1); j++) {
			fscanf(mf, "%lf", &mat_get(m, w, i, j));
		}
		fscanf(vf, "%lf", &mat_get(m, w, i, j));
	}
	fclose(mf);
	fclose(vf);

	*matrix = m;
	*width = w;
	*height = h;
}

int row_find_proc(int row_idx, int world_size, int *counts) {
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

void row_swap(
	int best_row, int curr_col, int *counts, int w,
	int world_size, double *subm, int subm_start, int global_rank,
	double *row_aux) {

	int proc_best = row_find_proc(best_row, world_size, counts);
	int proc_curr = row_find_proc(curr_col, world_size, counts);
	MPI_Status status;

	if (proc_best == proc_curr) {
		if (proc_best == global_rank) {
			int row1_idx = curr_col - subm_start;
			int row2_idx = best_row - subm_start;

			double *row1 = mat_row(subm, w, row1_idx);
			double *row2 = mat_row(subm, w, row2_idx);

			// swap rows
			memcpy(row_aux, row1, w*sizeof(*subm));
			memcpy(row1, row2, w*sizeof(*subm));
			memcpy(row2, row_aux, w*sizeof(*subm));
		}
	} else {
		if (global_rank == proc_best) { // process contains best row
			int row_idx = best_row - subm_start;
			double *subm_row = mat_row(subm, w, row_idx);
			MPI_Request req;

			MPI_Isend(
				subm_row,			// send buffer
				w, MPI_DOUBLE,		// type and number
				proc_curr,			// dest
				TAG_row_swap,		// tag
				MPI_COMM_WORLD,
				&req);
			MPI_Recv(
				row_aux,
				w, MPI_DOUBLE,
				proc_curr,
				TAG_row_swap,
				MPI_COMM_WORLD,
				&status);
			MPI_Wait(&req, &status);

			memcpy(subm_row, row_aux, w*sizeof(*subm));
		} else if (global_rank == proc_curr) {
			int row_idx = curr_col - subm_start;
			double *subm_row = mat_row(subm, w, row_idx);
			MPI_Request req;

			MPI_Isend(
				subm_row,			// send buffer
				w, MPI_DOUBLE,		// type and number
				proc_best,			// dest
				TAG_row_swap,		// tag
				MPI_COMM_WORLD,
				&req);
			MPI_Recv(
				row_aux,
				w, MPI_DOUBLE,
				proc_best,
				TAG_row_swap,
				MPI_COMM_WORLD,
				&status);
			MPI_Wait(&req, &status);

			memcpy(subm_row, row_aux, w*sizeof(*subm));
		}
	}
}

void row_normalize(double *row, int col, int w) {
	int i;
	double first;

	first = row[col];
	#pragma omp parallel for num_threads(8)
	for (i = 0; i < w; i++) {
		row[i] /= first;
	}
}

void row_elim_col(
	const double *row, double *dest_row, int w, int elim_col) {

	int i;
	double first;

	first = dest_row[elim_col];

	#pragma omp parallel for
	for (i = 0; i < w; i++) {
		dest_row[i] -= first * row[i];
	}
}