OUTFILE := ./out
NTFLAG :=

np ?= 4

ifneq (,$(nr))
$(shell cp matriz$(nr).txt matriz.txt)
$(shell cp vetor$(nr).txt vetor.txt)
endif

ifneq (,$(nt))
NTFLAG := -D NUM_THREADS_PER_PROCESS=$(nt)
endif

all:
	@mpicc $(NTFLAG) -fopenmp -lm -o $(OUTFILE) main.c

run:
	@mpirun --hostfile hosts -np $(np) $(OUTFILE)

clean:
	@rm $(OUTFILE)
