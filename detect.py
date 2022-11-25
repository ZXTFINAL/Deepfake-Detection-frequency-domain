import torch
from torch import nn
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import os
import dlib
from model import XCE4_Net
from data import dct
model = XCE4_Net()

model.load_state_dict(torch.load("pretrained.pth"))# 见 release
model.eval()

model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_result(imgpath):

    hog_face_detetor = dlib.get_frontal_face_detector()

    img = cv2.imread(imgpath)
    imgg3 = cv2.GaussianBlur(img,(3,3),0)
    imgg5 = cv2.GaussianBlur(img,(5,5),0)

    cv2.imwrite("origin.png", img)
    cv2.imwrite("gaussian3x3.png", imgg3)
    cv2.imwrite("gaussian5x5.png", imgg5)

    detections = hog_face_detetor(img,1)
    data1 = []
    data2 = []
    data3 = []
    imgs = []
    for face in detections:
        x = face.left()
        y = face.top()
        r = face.right()
        b = face.bottom()

        face_rect = img[np.maximum(y-10,0):b+10, np.maximum(x-10,0):r+10]


        img = cv2.resize(face_rect,(299,299))
        newimg = np.zeros_like(img)

        for m in range(3):
            imgx = cv2.resize(img[:,:,m],(299,299))
            f = np.fft.fft2(imgx)
            fshift = np.fft.fftshift(f)
            res = np.log(np.abs(fshift))

            newimg[:,:,m] = res
        f = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        fshift = np.fft.fftshift(f)
        res = np.log(np.abs(fshift))
        plt.subplot(1,5,2), plt.imshow(np.array(newimg, np.uint8))
        plt.axis('off')
        # plt.show()
        data1 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(img, dtype=np.float),(299,299)), [2,0,1])], dtype=np.float),device=device).float()
        data2 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(dct(img,4), dtype=np.float),(299,299)), [2,0,1])], dtype=np.float),device=device).float()
        data3 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(dct(img,5), dtype=np.float),(299,299)), [2,0,1])], dtype=np.float),device=device).float()
        data4 = torch.tensor(np.array([np.transpose(cv2.resize(np.array(newimg, dtype=np.float),(299,299)), [2,0,1])], dtype=np.float),device=device).float()

        plt.subplot(1,5,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title(str(nn.Softmax(dim=1)(model(data1, data2, data3,data4))[0][1].detach().cpu().numpy()))
        plt.axis('off')
        # plt.show()
        for i in range(3):

            plt.subplot(1,5,i+3), plt.imshow(cv2.resize(dct(img,i+3),(299,299)))
            plt.axis('off')
        # plt.show()
        print(nn.Softmax(dim=1)(model(data1, data2, data3,data4)))

    plt.savefig("output.png")
if __name__ == "__main__":
    plot_result("000002.png")
