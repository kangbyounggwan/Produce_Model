import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import pdb
from scipy.stats import skew, kurtosis
import time
import seaborn as sns
import os
import datetime
# ae_result,iso_result,LOF_result,scenario_len
def plot(rrcf_result,a_e_result,iso_result,lof_result,scenario_len):

    data = [rrcf_result,a_e_result,iso_result,lof_result]
    for i in range(len(data)):
        plt.figure(figsize=(16, 8))
        index = np.arange(len(data[i]))
        plt.bar(index, data[i])
        for j in scenario_len:
            plt.axvline(x = j, color='red', linestyle='dashed',label='25%',alpha=0.5)

    plt.show()