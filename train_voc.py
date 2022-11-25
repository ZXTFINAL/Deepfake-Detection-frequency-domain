
from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import init
import matplotlib.pyplot as plt
from data import get_data
from model import XCE4_Net

EPOCHS = 10
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
TRAIN_REAL_DATAPATH = "train/real"
TRAIN_FAKE_DATAPATH = "train/fake"
TEST_REAL_DATAPATH = "test/real"
TEST_FAKE_DATAPATH = "test/fake"



def build():
    model = XCE4_Net()
    print(model)
    # model.load_state_dict(torch.load("pretrained_detector.pth"))
    loss_fn = nn.CrossEntropyLoss()
    from torch import optim
    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def main():
    model, loss_fn, optimizer = build()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.cuda()
    for i in range(epochs):
        print("Epoch: {}".format(i+1))
        print("Training...")
        num = 0
        best_acc = 0.0
        for X_train1, X_train2, X_train3, X_train4, y_train, orimgs in tqdm(get_data(allreal=random.shuffle(os.listdir(TRAIN_REAL_DATAPATH)), allfake=random.shuffle(os.listdir(TRAIN_FAKE_DATAPATH)), batch_size=TRAIN_BATCH_SIZE)):
            num += 1
            model.train()
            optimizer.zero_grad()

            y_pred = model(X_train1, X_train2, X_train3, X_train4)
            loss = loss_fn(y_pred, y_train)
            print(loss)
            loss.backward()
            optimizer.step()
            if num % 200 == 199:
                print("Testing...")
                correct = 0
                alldata = 0
                for X_test1, X_test2, X_test3, X_test4, y_test, orimgs in tqdm(get_data(allreal=random.shuffle(os.listdir(TEST_REAL_DATAPATH)), allfake=random.shuffle(os.listdir(TEST_FAKE_DATAPATH)), batch_size=TEST_BATCH_SIZE)):
                    alldata += X_test1.size(0)
                    model.eval()
                    with torch.no_grad():
                        y_batch_pred = model(
                            X_test1, X_test2, X_test3, X_test4)
                        index, predicted = torch.max(y_batch_pred.data, axis=1)
                        correct += predicted.eq(
                            y_test.data.view_as(predicted)).sum()
                print("Test data acc: {}  {}/{}".format(correct /
                      alldata, correct, alldata))
                if correct/alldata > best_acc:
                    torch.save(model.state_dict(), "best_"+str(i) +
                               "_detector_%f.pth" % (correct/alldata))
                    best_acc = correct/alldata
                torch.save(model.state_dict(), "period_%03d_detector.pth" % i)
