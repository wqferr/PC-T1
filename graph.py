import pandas as pd
from matplotlib import pyplot as plt

def plot(filename, xlabel, ylabel):
	data = pd.read_csv(filename, index_col=False)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	x = data.iloc[:, 0]
	for col in data.columns[1:]:
		y = data[col]
		plt.plot(x, y, label=col)

	plt.legend()

if __name__ == '__main__':
	plots = [
		('times_nproc_1000rows', 'Number of processes'),
		('times_nproc_5000rows', 'Number of processes'),
		('times_nproc_10000rows', 'Number of processes'),

		('times_nrows_1proc', 'Number of rows'),
		('times_nrows_2proc', 'Number of rows'),
		('times_nrows_4proc', 'Number of rows'),
		('times_nrows_8proc', 'Number of rows'),
		('times_nrows', 'Number of rows')
	]

	for p in plots:
		plt.figure()
		plot(p[0] + '.txt', p[1], 'Response time (seconds)')
		plt.savefig(p[0] + '.png', bbox_inches='tight')