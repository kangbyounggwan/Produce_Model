import numpy as np
import pandas as pd
import os
from preprocessing import fft_t_sum
from test_model import svm_, lof, iso, AE_, rrcf
import glob
from plot import plot
import pdb
all_data = pd.DataFrame()
fft_sum = []
scenario_len = []
path = 'D:\\현대 다이나모 sc 주행 시나리오별\\'
a = glob.glob(path + '시나리오*'
                     '\\')
for folder in a:  # 예를들어 201901, 201902 로 된 파일이면 2019_*
    file = glob.glob(folder + '2022*')
    length =[]
    for i in range(len(file)):
        print(file[i])
        colums = ['1CH', '2CH', '3CH', '4CH', '5CH']
        data = pd.read_csv(file[i], names=colums, header=None)
        shape_point = int(len(data) / 8000)
        print(shape_point,'초')
        data = data[:shape_point*8000]

        one_ch = data['1CH'].values.reshape(shape_point, 8000)
        print(one_ch.shape)
        length.append(len(one_ch[int(len(one_ch) / 2):int(len(one_ch) / 2) + 30]))
        fft_sum = fft_t_sum(one_ch[int(len(one_ch) / 2):int(len(one_ch) / 2) + 30])
        all_data = all_data.append(fft_sum,ignore_index=True)
        print(all_data)
        if len(file)-1 == i:
            if scenario_len == []:
                scenario_len.append(sum(length))
            else:
                scenario_len.append(scenario_len[-1]+sum(length))

print(scenario_len)

print(all_data)
# SVM_result = svm_(all_data)
LOF_result = lof(all_data)
iso_result = iso(all_data)
ea_result = AE_(all_data)
rrcf_result = rrcf(all_data)
plot(rrcf_result,ea_result,iso_result,LOF_result,scenario_len)


# print('isolation',iso_result)
# print('local outlier factor',LOF_result)
# print(SVM_result)
