#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include <mpi.h>

#define mat_get(m, w, i, j) (m[w*i + j])

void read_matrix(double **m, int *w, int *h);

enum {
	TAG_PING
};

int main(int argc, char *argv[]) {
	int global_rank, world_size;
	int i, j;
	int w, h; // matrix dimensions
	int *displs = NULL; // displacements for MPI_Scatterv
	int *counts = NULL; // counts for MPI_Scatterv
	double *m = NULL; // the entire matrix
	double *subm = NULL; // rows this process is responsible for
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	displs = calloc(world_size, sizeof(*displs));
	counts = malloc(world_size * sizeof(*counts));

	if (global_rank == 0) {
		int rows_per_process;	// min number of rows per process
								// remaining rows are distributed to processes
								// with lowest ranks

		read_matrix(&m, &w, &h);

		rows_per_process = h / world_size;

		// set min rows per process
		for (i = 0; i < world_size; i++) {
			counts[i] = rows_per_process * w;
		}
		// add single row to process so as to distribute load
		for (i = 0; i < h % world_size; i++) { // distribute remaining rows
			counts[i] += w;
		}

		// set displacements based on cumulative sum of counts
		for (i = 1; i < world_size; i++) {
			displs[i] = displs[i-1] + counts[i-1];
		}
	}

	// broadcast matrix dimensions
	MPI_Bcast(
		&w,
		1, MPI_INT,
		0,
		MPI_COMM_WORLD);
	MPI_Bcast(
		&h,
		1, MPI_INT,
		0,
		MPI_COMM_WORLD);

	// broadcast number of rows each process is responsible for
	MPI_Bcast(
		counts, 		// buffer
		world_size, MPI_INT, // block description
		0,				// root
		MPI_COMM_WORLD);
	// allocate submatrix given number of rows and size
	subm = malloc(counts[global_rank] * sizeof(*subm));

	MPI_Scatterv(
		m,				// send buffer
		counts, displs,	// send block size and displacement
		MPI_DOUBLE,		// send type
		subm,			// receive buffer
		counts[global_rank], // receive count
		MPI_DOUBLE,		// receive type
		0,				// root
		MPI_COMM_WORLD);

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
	for (i = 0; i < counts[global_rank]; i++) {
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