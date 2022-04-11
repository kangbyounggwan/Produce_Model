import torch
import torch.nn as nn

import numpy as np
import joblib
import os
import pdb

from random import sample
# import matplotlib.pyplot as plt
from ae_model import AutoEncoder

basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE_PATH = basedir + '/ModelFiles/'


AE_loss = nn.MSELoss()



def train(model, Loss, optimizer, num_epochs,train_loader,device):
    print(num_epochs)
    print('a')
    train_loss_arr = []
    test_loss_arr = []
    best_epoch = 0
    best_test_loss = 9999999999999999
    early_stop, early_stop_max = 0., 10000.

    for epoch in range(num_epochs):
        # print(epoch)

        epoch_loss = 0.
        for batch_X in train_loader:
            batch_X = batch_X.type(torch.Tensor).to(device)
            optimizer.zero_grad()

            # Forward Pass
            model.train()
            outputs = model(batch_X)
            train_loss = Loss(outputs, batch_X)
            # print('정답',batch_X[0])
            # print('예측',outputs[0])
            # Backward and optimize
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
        train_loss_arr.append(epoch_loss / len(train_loader.dataset))

        if epoch % 10 == 0:
            model.eval()
            test_loss = 0.
            outputs = 0
            batch_X =0
            for batch_X in train_loader:
                batch_X = batch_X.type(torch.Tensor).to(device)

                # Forward Pass
                outputs = model(batch_X)
                batch_loss = Loss(outputs, batch_X)


                test_loss += batch_loss.item()
            print('정답',batch_X[0])
            print('예측',outputs[0])
            test_loss = test_loss
            test_loss_arr.append(test_loss)
            if best_test_loss >= test_loss:
                best_test_loss = test_loss
                early_stop = 0
                best_epoch = epoch

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss / len(train_loader.dataset),
                                                                                      test_loss))
            else:
                early_stop += 1
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss / len(train_loader.dataset),
                                                                                    test_loss))


        if early_stop >= early_stop_max:
            break

    current_model_path = MODEL_FILE_PATH + format('ae[{}].model'.format(best_epoch))
    model_write_file = open(current_model_path, "wb")
    joblib.dump(model, model_write_file)
    model_write_file.close()