OUTFILE := ./out

np?=4

all:
	@mpicc -Wall -fopenmp -lm -o $(OUTFILE) main.c

dbg:
	@mpicc -fopenmp -lm -g -o $(OUTFILE) main.c

run:
	@mpirun -np $(np) $(OUTFILE)

runv:
	@valgrind --leak-check=full --show-leak-kinds=all mpirun -np $(np) $(OUTFILE)

clean:
	@rm $(OUTFILE)