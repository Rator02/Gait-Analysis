import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

def read_data(file_path, columns):
	'''
	read files according to file_path and columns
	args:
		file_path, columns (list)
	return:
		data (numpy array)
	'''
	if not os.path.isfile(file_path):
		raise AssertionError(file_path, 'not found')
	data = pd.read_csv(file_path)
	return data


def init_samplingRate (file_path):

	if not os.path.isfile(file_path):
		raise AssertionError(file_path, 'not found')
	mode='r'
	with open (file_path, mode) as f:
		lines=f.readlines()
		duration = lines[-1].rstrip().split(',')
		sf_time = duration[0]
		num_rows=len(lines)
		print(str(sf_time), 'TIME!')
		fs = int((len(lines)-1)/float(sf_time))
		print(str(fs), 'SAMPLING FREQUENCY!')
		return fs

def plot_lines(data, fs, plot_title):
	num_rows, num_cols=data.shape
	if num_cols!=4:
		raise ValueError('Not 3D data')
	data.plot(x = 'Time (s)')
	plt.legend(loc = 'best', title = plot_title)
	#data.plot.kde()


def butt_filter(data, cutoff, f_sample):
	f_data=data.copy()
	f_data = f_data.transpose()
	sos = signal.butter(5, cutoff, 'low', fs = f_sample, output = 'sos')
	for i in range(len(data.columns)):
			if i > 1:
				f_data.iloc[i] = signal.sosfilt(sos, f_data.iloc[i])
	f_data.info()
	f_data = f_data.transpose()
	f_data.info()
	return f_data
