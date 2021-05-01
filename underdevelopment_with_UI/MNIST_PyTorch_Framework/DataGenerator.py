import os
import numpy as np
from PIL import Image
import csv
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import re
import cv2
import imageio
import PIL
import math
import torch
import pdb
from torchvision.transforms import functional as F

def loadNpy(inpath):
    return np.load(inpath)

class DefaultDataGenerator(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile):
        df =  pd.read_csv(pathDatasetFile, sep = ',', names = ['inp_pth', 'out_pth'], header = None)
        self.input_imgpath = df.iloc[:,0].values.tolist()
        self.out_labpth = df.iloc[:,1].values.tolist()

    def __getitem__(self, index):
        try:
            img_pth = self.input_imgpath[index]
            labpth = self.out_labpth[index]
            img = loadNpy(img_pth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (1,img.shape[0],img.shape[1]))
            output = loadNpy(labpth)
            #out = np.zeros(10)
            #out[int(output[0])] = 1.0
            out = output
            #pdb.set_trace()
            #print(img.shape, out.shape)

        except:
            print('Not Found')
            img = np.zeros((28,28))
            out = 100
        img = torch.from_numpy(img).float()
        out = torch.from_numpy(out).float()
        return img, out

    def __len__(self):
        return len(self.input_imgpath)