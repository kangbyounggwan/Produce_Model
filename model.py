import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rrcforest import RobustRandomCutForest
import torch.nn as nn
import torch.optim as optim
import torch
from ae_train import train
from ae_model import AutoEncoder
import pickle
from sklearn.externals import joblib

basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'
import wandb
model_spot = 'MSC_주행'
def SVM(fft_sum):
    fft_sum = np.array(fft_sum)
    svm = sklearn.svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)

    svm.fit(fft_sum)
    score = -svm.score_samples(fft_sum)
    predict = svm.predict(fft_sum)
    svm_score = score.tolist()
    svm_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format(model_spot + '_svm.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(iso, model_write_file)
    model_write_file.close()

    return print('success new svm_model')


def LOF(fft_sum):
    fft_sum = np.array(fft_sum)
    lof = LocalOutlierFactor(n_neighbors=1000, algorithm='ball_tree', novelty=True, contamination=0.0000001,
                             n_jobs=-1)
    lof.fit(fft_sum)
    score = -lof.score_samples(fft_sum)
    predict = lof.predict(fft_sum)
    lof_score = score.tolist()
    lof_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format(model_spot +'_lof.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(lof, model_write_file)
    model_write_file.close()

    return print('success new lof_model')

def rrcf(fft_sum):
    fft_sum = np.array(fft_sum)
    rrcf = RobustRandomCutForest(n_estimators=100, n_jobs=-1, contamination=0.0000001)
    print(fft_sum)

    rrcf.fit(fft_sum)
    score = rrcf.log_codisp_samples(fft_sum,fft_sum)
    predict = rrcf.predict(fft_sum)
    lof_score = score.tolist()
    lof_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format(model_spot +'_rrcf.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(rrcf, model_write_file)
    model_write_file.close()
    print(score)

    return print('success new rrcf_model')

def iso(fft_sum):
    fft_sum = np.array(fft_sum)
    iso = IsolationForest(n_estimators=100, contamination=0.0000001,
                          n_jobs=-1)
    iso.fit(fft_sum)
    score = -iso.score_samples(fft_sum)
    predict = iso.predict(fft_sum)
    iso_score = score.tolist()
    iso_predict = predict.tolist()

    current_model_path = MODEL_FILE_PATH + format(model_spot + '_iso.model')
    model_write_file = open(current_model_path, "wb")
    joblib.dump(iso, model_write_file)
    model_write_file.close()

    return print('success new iso_model')


def autoencoder(fft_sum):
    wandb.init(project="hyundai-learningrate_0.001 ", entity="kbg_")

    print(fft_sum)
    fft_sum_ = np.array(fft_sum)
    print(fft_sum_)

    batch_size = 100
    num_epochs = 10000
    learning_rate = 0.001
    train_loader = torch.utils.data.DataLoader(dataset=fft_sum_, batch_size=batch_size, shuffle=False)
    print(train_loader.dataset[0][0])
    AE_loss = nn.MSELoss()
    device = torch.device('cpu')
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }
    AE = AutoEncoder(7, 6, 5, 4)
    AE = AE.to(device)
    AE_optimizer = optim.Adam(AE.parameters(), lr=learning_rate, weight_decay=1e-4)
    train(AE, AE_loss, AE_optimizer, num_epochs, train_loader, device, model_spot)

    return print('success new ae_model')
