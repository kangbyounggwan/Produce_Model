import glob, os
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import math
import pdb


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
    # fft 변환
    for i in range(len(raw)):
        raw_ = np.array(raw[i])
        fs = 8000
        optimiz = raw_ / fs

        frequency_raw = np.fft.rfft(optimiz, n=8000)
        # rms 변환
        rms_result = rms(raw[i])
        # 값 저장
        p_to_p = max(raw[i]) - min(raw[i])
        std = np.std(frequency_raw)
        mean = np.mean(frequency_raw)
        skew_ = skew(frequency_raw)
        var = np.var(frequency_raw)
        kurt = kurtosis(frequency_raw)
        col_p_to_p.append(float(p_to_p))
        col_rms.append(float(rms_result))
        col_std.append(float(std))
        col_mean.append(float(abs(mean)))
        col_skew.append(float(abs(skew_)))
        col_var.append(float(abs(var)))
        col_kurt.append(np.round(abs(kurt), 0))
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
    print(t_sum_raw)
    numpy_ = np.array(t_sum_raw)
    print(numpy_)
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
