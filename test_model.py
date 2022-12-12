import pandas as pd
import numpy as np
import joblib
import os
import torch.nn as nn
import torch
import time
from rrcforest import RobustRandomCutForest

basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'


def rrcf(t_sum_raw):
    model_data = MODEL_FILE_PATH + 'FFT_RRCF_TRAIN_SC_주행.csv'
    train_data = pd.read_csv(model_data)

    data = pd.DataFrame(t_sum_raw)
    rrcf = RobustRandomCutForest(n_estimators=100, n_jobs=-1, contamination=0.0000001)
    rrcf.fit(train_data)
    rrcf_result = rrcf.predict(data)
    data_start = time.time()
    rrcf_score = rrcf.codisp_samples(data, train_data)
    print("RRCF_inference_time :", time.time() - data_start, 's')
    return rrcf_score


def lof(t_sum_raw):

    data = np.array(t_sum_raw)
    path = MODEL_FILE_PATH + 'FFT_SC_주행_lof.model'
    print(data)
    with open(path, "rb") as file:
        model = joblib.load(file)

    lof_result = model.predict(data)
    data_start = time.time()
    lof_score = - model.score_samples(data)

    print("LOF_inference_time :", time.time() - data_start, 's')
    return lof_score


def svm_(t_sum_raw):
    data = np.array(t_sum_raw)
    path = MODEL_FILE_PATH + 'FFT_SC_주행_svm.model'
    print(data)
    with open(path, "rb") as file:
        model = joblib.load(file)

    svm_result = model.predict(data)
    data_start = time.time()
    svm_score = - model.score_samples(data)
    print("SVM_inference_time :", time.time() - data_start, 's')
    return svm_score


def iso(t_sum_raw):

    data = np.array(t_sum_raw)
    path = MODEL_FILE_PATH + 'FFT_SC_주행_iso.model'
    # print(data)
    with open(path, "rb") as file:
        model = joblib.load(file)

    iso_result = model.predict(data)
    data_start = time.time()
    iso_score = - model.score_samples(data)
    print("iso_inference_time:", time.time() - data_start, 's')
    return iso_score


def AE_(t_sum_raw):

    data = np.array(t_sum_raw)
    data = torch.Tensor(data)
    path = MODEL_FILE_PATH + 'FFT_SC_주행_AE_model.pt'
    # print(data)
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    loss = nn.MSELoss()
    ae_score = []
    data_start = time.time()
    output = model(data)
    print(output)
    print(len(output))
    for i in range(len(data)):
        ae_scores = loss(data[i], output[i])
        ae_scores = ae_scores.item()
        ae_score.append(ae_scores)
    print("A-E_inference_time:", time.time() - data_start, 's')
    return ae_score
