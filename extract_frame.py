####################################

# extract_frame

# Function to cut the data to a fixed number of data points.
# The data points represent the number of seconds of gait data.
# We have selected this to be 8s as one of the measurement criteria was to have at least 10s of gait data.
# Considering that slowest walking gait is 0.5 Hz (under which it seldom is), we get still get 4 cycles of gait data.


# Signal input is in form of a Pandas dataframe and integer
# Returns the acceleration and gyro signal with data points equal to 8s of gait data, and data points in the frame

####################################

# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.fft import fft, ifft
# import glob
# import re
# import pandas as pd
# import numpy as np

def extract_frame(acc_cut, gyr_cut, smp_frq):
  
    data_points = (smp_frq*8)    # Change the number in this line to change the timeframe of the data
    
    # '-1' as indexing starts from 0  
    acc_frame = acc_cut.iloc[0: data_points - 1, :]
    gyr_frame = gyr_cut.iloc[0: data_points - 1, :]
    
    return gyr_frame, acc_frame, data_points