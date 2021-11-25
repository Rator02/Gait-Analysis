####################################

# cut_data

# Function to cut the data to remove initial junk signal
# and then cut the data into a standard frame to feed into the ANN

# Signal input is in form of a Pandas dataframe and integer
# Returns the acceleration and gyroscope signal

####################################

# import matplotlib.pyplot as plt
# from scipy.fft import fft, ifft
# import glob
# import re
# import pandas as pd

import numpy as np
from scipy import signal

def cut_data(acc_flt, gyr_flt, smp_frq):
    
    # SJ: Denny had based his filter on the absolute acceleration.
    # But he mentioned saying that using the angular velocity is better.
    # Changing the code to based it on the angular velocity does result in a better clean up
    # of the unwanted segment at the start and end of the signal

    gyr_flt_np = gyr_flt.to_numpy()
    gyr_flt_np = gyr_flt_np[:,1:4]    # To remove the time column
    gyr_abs = np.linalg.norm(gyr_flt_np, axis=1)    # To obtain the RMS of the angular velocities
    

    peaks, _ = signal.find_peaks(gyr_abs, height=(1.1*np.mean(gyr_abs)), distance=smp_frq/2)

    diff_peaks = np.diff(peaks)

    gap1 = np.argmax(diff_peaks[:20])

    gap2 = np.argmax(diff_peaks[-10:])              # These two lines can be removed as our signal would most likely
    gap2 = int(gap2 + np.shape(diff_peaks) - 10)    # be cut before it reached the junk at the end

    acc_cut = acc_flt.iloc[peaks[gap1 + 1]:peaks[gap2], :]
    gyr_cut = gyr_flt.iloc[peaks[gap1 + 1]:peaks[gap2], :]

    # Choosing the next peak as the first gait cycle seems to have a lower amplitude than other cycles
    
    
    gyr_flt_np = gyr_cut.to_numpy()
    gyr_flt_np = gyr_flt_np[:,1:4]    # To remove the time column
    gyr_abs = np.linalg.norm(gyr_flt_np, axis=1)    # To obtain the RMS of the angular velocities
    peaks, _ = signal.find_peaks(gyr_abs, height=0, distance=smp_frq/2)    # Height is zero here as we need
                                                                           # the very next peak

    acc_cut = acc_cut.iloc[peaks[1]:,:]
    gyr_cut = gyr_cut.iloc[peaks[1]:,:]
    
    
    return acc_cut, gyr_cut