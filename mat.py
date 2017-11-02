import numpy as np

from numpy.random import rand
from sys import argv

def get_arg(idx, default):
	if len(argv) > idx:
		return argv[idx]
	return default

if __name__ == '__main__':
	n_rows = int(get_arg(1, '100'))
	suffix = get_arg(2, '')

	m_fname = 'matriz{}.txt'.format(suffix)
	v_fname = 'vetor{}.txt'.format(suffix)
	num_fmt = '%.3f'

	m = rand(n_rows, n_rows)
	v = rand(n_rows, 1)

	np.savetxt(m_fname, m, fmt=num_fmt)
	np.savetxt(v_fname, v, fmt=num_fmt)