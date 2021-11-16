import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import*

def test_data(file_name):
	cur_dir=os.getcwd()
	cutoff=10
	file_path=os.path.join(cur_dir, 'data', file_name)
	data=read_data(file_path, [1,2,3])
	data.describe()
	fs = init_samplingRate(file_path)
	#fs = 600
	plot_lines(data, fs, 'Raw data')
	butterworth_data = butt_filter(data, cutoff, fs)
	plot_lines(butterworth_data, fs, 'Butterworth Filter, cutoff: 15Hz')
	plt.show()

if __name__ == '__main__':
	if len(sys.argv)<2:
		raise ValueError('No file name specified')
	test_data(sys.argv[1])
