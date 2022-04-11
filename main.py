import numpy as np
import pandas as pd
import os
from preprocessing import fft_t_sum
from model import SVM, LOF, iso, autoencoder

path_to_dir = 'C:\\Users\\ST200423\\Desktop\\진동data 개발\\FREQUENCY\\dataset\\data'
path_to_test = os.path.join(path_to_dir, 'nomal.csv')

colums = ['1CH', '2CH', '3CH', '4CH', '5CH', '6CH', '7CH', '8CH', 'Time_Stamp']
data = pd.read_csv(path_to_test, names=colums, header=None)

one_ch = data['1CH'].values.reshape(-1, 8000)

fft_sum = fft_t_sum(one_ch)

SVM_result = SVM(fft_sum)
LOF_result = LOF(fft_sum)
iso_result = iso(fft_sum)
ea_result = autoencoder(fft_sum)
