OUTFILE := ./out

np?=4

all:
	@mpicc -fopenmp -o $(OUTFILE) main.c

dbg:
	@mpicc -fopenmp -g -o $(OUTFILE) main.c

run:
	@mpirun -np $(np) $(OUTFILE)

runv:
	@valgrind --leak-check=full mpirun -np $(np) $(OUTFILE)

clean:
	@rm $(OUTFILE)