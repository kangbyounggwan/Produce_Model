import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

import torch.nn as nn
import torch.optim as optim
import torch
from ae_train import train
from ae_model import AutoEncoder

basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'


def SVM(fft_sum):
    fft_sum = np.array(fft_sum, dtype='complex_').astype(float)
    svm = sklearn.svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)

    svm.fit(fft_sum)
    score = -svm.score_samples(fft_sum)
    predict = svm.predict(fft_sum)
    svm_score = score.tolist()
    svm_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format('svm.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(iso, model_write_file)
    model_write_file.close()

    return print('success new svm_model')


def LOF(fft_sum):
    fft_sum = np.array(fft_sum, dtype='complex_').astype(float)
    lof = LocalOutlierFactor(n_neighbors=1000, algorithm='ball_tree', novelty=True, contamination=0.0000001,
                             n_jobs=-1)
    lof.fit(fft_sum)
    score = -lof.score_samples(fft_sum)
    predict = lof.predict(fft_sum)
    lof_score = score.tolist()
    lof_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format('lof.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(lof, model_write_file)
    model_write_file.close()

    return print('success new lof_model')


def iso(fft_sum):
    fft_sum = np.array(fft_sum, dtype='complex_').astype(float)
    iso = IsolationForest(n_estimators=100, contamination=0.0000001,
                          n_jobs=-1)
    iso.fit(fft_sum)
    score = -iso.score_samples(fft_sum)
    predict = iso.predict(fft_sum)
    iso_score = score.tolist()
    iso_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format('iso.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(iso, model_write_file)
    model_write_file.close()

    return print('success new iso_model')


def autoencoder(fft_sum):
    print(fft_sum)
    fft_sum_ = np.array(fft_sum)
    print(fft_sum_)

    batch_size = 100
    num_epochs = 100000
    learning_rate = 0.0001
    train_loader = torch.utils.data.DataLoader(dataset=fft_sum_, batch_size=batch_size, shuffle=False)
    print(train_loader.dataset[0][0])
    AE_loss = nn.MSELoss()
    device = torch.device('cpu')

    AE = AutoEncoder(7, 6, 5, 4, 3, 2)
    AE = AE.to(device)

    AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
    train(AE, AE_loss, AE_optimizer, num_epochs, train_loader, device)

    return print('success new ae_model')
