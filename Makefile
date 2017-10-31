OUTFILE := ./out

np?=4

all:
	@mpicc -fopenmp -o $(OUTFILE) main.c

run:
	@mpirun -np $(np) $(OUTFILE)

clean:
	@rm $(OUTFILE)