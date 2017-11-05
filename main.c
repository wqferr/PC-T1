#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <mpi.h>

#define mat_get(m, w, i, j) ((m)[(i)*(w) + (j)])
#define mat_row(m, w, i) (&((m)[(i)*(w)]))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define MATRIX_FILENAME "matriz.txt"
#define VECTOR_FILENAME "vetor.txt"
#define OUT_FILENAME "resultado.txt"

#ifndef NUM_THREADS_PER_PROCESS
#define NUM_THREADS_PER_PROCESS 8
#endif

void read_matrix(double **m, int *w, int *h);
int row_find_proc(int row_idx, int world_size, int *proc_row_count);
void row_normalize(double *row, int col, int w);
void row_elim_col(
	const double *row, double *dest_row, int w, int elim_col);

// Tags used for messages.
// Only 1:1 communication in the algorithm is for row swapping
enum {
	TAG_ROW_SWAP
} msg_tag;

// Used for reduction with MAXLOC
// absval is the absolute value of the corresponding position
// row is the index of its row
typedef struct {
	double absval;
	int row;
} column_element;

int main(int argc, char *argv[]) {
	int global_rank, world_size;
	int i;
	int subm_start; // first row the process is responsible for
	int subm_n_rows; // number of rows of the process's submatrix
	int elim_idx = 0; // current row being eliminated
	int w, h; // matrix dimensions
	int max_elim_col; // the number of iterations the algorithm should do
	double t; // for measuring response time
	int *displs = NULL; // displacements for MPI_Scatterv
	int *proc_row_count = NULL; // number of rows each process is responsible for
	int *proc_elm_count = NULL; // proc_row_count for MPI_Scatterv
	double *m = NULL; // the entire matrix
	double *subm = NULL; // rows this process is responsible for
	double *elim_row = NULL; // row to be used for elimination
	MPI_Status status;
	FILE *of; // output file of result vector

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	displs = calloc(world_size, sizeof(*displs));
	proc_row_count = malloc(world_size * sizeof(*proc_row_count));
	proc_elm_count = malloc(world_size * sizeof(*proc_elm_count));

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
			proc_row_count[i] = rows_per_process;
			proc_elm_count[i] = w*rows_per_process;
		}
		// add single row to process so as to distribute load
		for (i = 0; i < h % world_size; i++) { // distribute remaining rows
			proc_row_count[i]++;
			proc_elm_count[i] += w;
		}

		// set arrays for MPI_Scatterv
		for (i = 1; i < world_size; i++) {
			displs[i] = displs[i-1] + proc_row_count[i-1]*w;
		}

		t = omp_get_wtime();
	}

	// ========== Broadcast information to other processes =========
	// height is only necessary for the root process at the end of execution

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
		proc_row_count, 		// buffer
		world_size, MPI_INT, // block description
		0,				// root
		MPI_COMM_WORLD);


	// find the index of the submatrix's first row
	subm_start = 0;
	for (i = 0; i < global_rank; i++) {
		// sum of the length of all the previous submatrices
		subm_start += proc_row_count[i];
	}

	// allocate submatrix given number of elements
	subm = malloc(proc_row_count[global_rank]*w * sizeof(*subm));
	subm_n_rows = proc_row_count[global_rank];

	MPI_Scatterv(
		m,				// send buffer
		proc_elm_count, displs,	// send block size and displacement
		MPI_DOUBLE,		// send type
		subm,			// receive buffer
		proc_row_count[global_rank]*w,	// receive count
		MPI_DOUBLE,		// receive type
		0,				// root
		MPI_COMM_WORLD);

	// ======== Begin processing =======
	elim_row = malloc(w * sizeof(*elim_row));

	// for every column in the matrix
	// the iterations are synchronized across all processes
	for (elim_idx = 0; elim_idx < max_elim_col; elim_idx++) {
		column_element best_local, best;
		int best_row_proc, elim_row_proc;

		best_local.absval = -1;
		best.absval = -1;

		// no sense in parallelizing this, all of the code
		// would be inside a critical region
		for (i = 0; i < subm_n_rows; i++) {
			if (i + subm_start >= elim_idx) {
				double v = fabs(mat_get(subm, w, i, elim_idx));
				if (v > best_local.absval) {
					best_local.absval = v;
					best_local.row = subm_start + i;
				}
			}
		}

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

		best_row_proc = row_find_proc(best.row, world_size, proc_row_count);
		elim_row_proc = row_find_proc(elim_idx, world_size, proc_row_count);
		if (global_rank == best_row_proc) {
			double *subm_row = mat_row(subm, w, best.row - subm_start);
			row_normalize(subm_row, elim_idx, w);
			memcpy(elim_row, subm_row, w*sizeof(*subm));
		}

		MPI_Bcast(
			elim_row,
			w, MPI_DOUBLE,
			best_row_proc,
			MPI_COMM_WORLD);

		if (best.row != elim_idx) { // swap required
			if (best_row_proc == elim_row_proc) { // intra-process swap
				if (global_rank == best_row_proc) {
					double *subm_row1 = mat_row(subm, w, best.row - subm_start);
					double *subm_row2 = mat_row(subm, w, elim_idx - subm_start);
					memcpy(subm_row1, subm_row2, w*sizeof(*subm_row1));
					memcpy(subm_row2, elim_row, w*sizeof(*subm_row2));
				}
			} else if (global_rank == best_row_proc) {
				double *subm_row = mat_row(subm, w, best.row - subm_start);
				MPI_Recv(
					subm_row,
					w, MPI_DOUBLE,
					elim_row_proc,
					TAG_ROW_SWAP,
					MPI_COMM_WORLD,
					&status);
			} else if (global_rank == elim_row_proc) {
				double *subm_row = mat_row(subm, w, elim_idx - subm_start);
				MPI_Send(
					subm_row,
					w, MPI_DOUBLE,
					best_row_proc,
					TAG_ROW_SWAP,
					MPI_COMM_WORLD);
				memcpy(subm_row, elim_row, w*sizeof(*subm_row));
			}
		}

		#pragma omp parallel for\
					num_threads(NUM_THREADS_PER_PROCESS)
		for (i = 0; i < subm_n_rows; i++) {
			if (i + subm_start != elim_idx) {
				row_elim_col(elim_row, mat_row(subm, w, i), w, elim_idx);
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Gatherv(
		subm,		// send buf
		w*proc_row_count[global_rank],	// send count
		MPI_DOUBLE,	// send type
		m,			// recv buf
		proc_elm_count,
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

	free(proc_elm_count);
	free(proc_row_count);
	free(displs);
	free(elim_row);
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

int row_find_proc(int row_idx, int world_size, int *proc_row_count) {
	int i;

	for (i = 0; i < world_size; i++) {
		if (row_idx < proc_row_count[i]) {
			return i;
		}

		// get the index relative to the next process
		row_idx -= proc_row_count[i];
	}

	return -1;
}

void row_normalize(double *row, int col, int w) {
	int i;
	double first;

	first = row[col];
	#pragma omp parallel for\
				num_threads(NUM_THREADS_PER_PROCESS)
	for (i = 0; i < w; i++) {
		row[i] /= first;
	}
}

void row_elim_col(
	const double *row, double *dest_row, int w, int elim_col) {

	int i;
	double first;

	first = dest_row[elim_col];

	// can't use omp parallel here, as this function call
	// is already nested into a parallel for loop
	for (i = 0; i < w; i++) {
		dest_row[i] -= first * row[i];
	}
}