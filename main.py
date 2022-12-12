import numpy as np
import pandas as pd
import os
from preprocessing import fft_t_sum
from model import SVM, LOF, iso, autoencoder,rrcf
import glob
import pdb
import wandb


basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'

all_data = pd.DataFrame()
fft_sum = []
a = glob.glob(r'E:\\download_secwh.hyundai-autoever.com(2022-05-27)\\시나리오1_MSC주행\\2022*.csv')
print(a)
for f in a:  # 예를들어 201901, 201902 로 된 파일이면 2019_*
    print(f)

    colums = ['1CH', '2CH', '3CH', '4CH', '5CH']
    data = pd.read_csv(f, names=colums, header=None)
    # print(data)
    shape_point = int(len(data) / 80000)
    print(shape_point, '초')
    data = data[:shape_point * 80000]
    one_ch = data['1CH'].values.reshape(shape_point, 80000)
    print(one_ch.shape)
    fft_sum = fft_t_sum(one_ch)
    all_data = all_data.append(fft_sum,ignore_index=True)
    print('data 취합',all_data)

all_data.to_csv('FFT_RRCF_TRAIN_MSC_주행.csv', index=False)

rrcf_result = rrcf(all_data)
# SVM_result = SVM(all_data)
# LOF_result = LOF(all_data)
# iso_result = iso(all_data)
# ea_result = autoencoder(all_data)

