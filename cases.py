from subprocess import run
from itertools import product
from sys import stdout

mat_num_rows = [1000, 5000, 10000]
num_proc = [1, 2, 4, 8]
num_thrd = [4, 8]

if __name__ == '__main__':
	print("nrows", "nproc", "nthrd", "time", sep='\t')
	for (nr, np, nt) in product(mat_num_rows, num_proc, num_thrd):
		print(nr, np, nt, sep='\t', end='\t')
		stdout.flush()
		run([
			"make", "all", "run",
			"nr={}".format(nr),
			"np={}".format(np),
			"nt={}".format(nt)])