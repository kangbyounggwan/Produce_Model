import glob, os
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import math
import pdb
import matplotlib.pyplot as plt

# 다변량 추출
def fft_t_sum(raw):
    col_p_to_p = []
    col_rms = []
    col_std = []
    col_mean = []
    col_skew = []
    col_var = []
    col_kurt = []
    col_len = []
    for i in range(len(raw)):

        fmax = 16000  # sampling frequency 1000 Hz
        dt = 0.01       # sampling period
        N = 16000    # length of signal

        # Fourier spectrum
        xf = np.fft.fft(raw[i])
        frequency_raw =np.abs(xf[5:int(N / 2)])

        # plt.figure(figsize=(16, 8))
        # plt.plot(frequency_raw)
        # plt.show()
        # rms 변환
        rms_result = rms(raw[i])
        # 값 저장
        p_to_p = max(raw[i]) - min(raw[i])
        std = np.std(frequency_raw)
        mean = np.mean(frequency_raw)
        skew_ = skew(frequency_raw)
        var = np.var(frequency_raw)
        kurt = kurtosis(frequency_raw)
        col_p_to_p.append(p_to_p)
        col_rms.append(rms_result)
        col_std.append(std)
        col_mean.append(mean)
        col_skew.append(skew_)
        col_var.append(var)
        col_kurt.append(kurt)
        col_len.append(len(frequency_raw))

    # pandas t_sum 정리
    t_sum_raw = pd.DataFrame({
        'p_to_p': col_p_to_p,
        'rms': col_rms,
        'Std': col_std,
        'Mean': col_mean,
        'Skew': col_skew,
        # 'Var': col_var,
        'kurt': col_kurt,
        'len': col_len,
    })
    return t_sum_raw


# rms 값 확인
def rms(data):
    squre = 0.0
    root = 0.0
    mean = 0.0

    # Calculating squre
    for i in range(len(data)):
        squre += (data[i] ** 2)
    # Calculating Mean
    mean = squre / len(data)
    # Calculating Root
    rms_result = math.sqrt(mean)

    return rms_result
