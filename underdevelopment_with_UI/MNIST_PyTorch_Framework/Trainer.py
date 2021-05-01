import os
import numpy as np
import time
import sys
import re
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as func
import csv
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
from DataGenerator import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import shutil
import pandas as pd
import torchvision.models as models
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from Model import *

class TrainerParams():
    def __init__(self, nnArchitecture, nnIsTrained = True, nnClassCount = 10,
                 trBatchSize = 16, trMaxEpoch = 50, optimizername = 'ADAM', scheduler = None,
                 lossname = 'default', evalloss = 'default', datageneratortype = 'default',
                 cuda = False, checkpoint = None, drop_rate = 0, t_0 = 0, t_mult = 1, first_index = 1,
                 out_activation = None):
        self.params = {
        'arch':nnArchitecture,
        'useimagenetweights':nnIsTrained,
        'num_classes': nnClassCount,
        'batch_size': trBatchSize,
        'num_epochs': trMaxEpoch,
        'scheduler': scheduler,
        'lossname':lossname,
        'evalloss':evalloss,
        'checkpoint':checkpoint,
        'dropout':drop_rate,
        'datageneratortype':datageneratortype,
        'optimizername':optimizername,
        'cuda':cuda,
        't_0':t_0,
        't_mult':t_mult,
        'first_index':first_index,
        'out_activation':out_activation}

    def getparams(self):
        return self.params

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val= 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n= 1):
        self.val = val
        self.sum+=val*n
        self.count+=n
        self.avg =self.sum/self.count


class BaseTrainer():
    def getOptimizer(self, model, optimizername):
        optimizer = optim.Adam(model.parameters(), lr = 1e-4, betas= (0.9,0.999), eps = 1e-08, weight_decay= 1e-5)
        return optimizer

    def getModel(self, params, device):

        model = Net()
        return model
    def getScheduler(self,optimizer, scheduler_type, t_max = 0, factor = 0.1, patience = 5):
        if scheduler_type == 'cos_anneal':
            return CosineAnnealingLR(optimizer, T_max = t_max)
        else:
            return ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')


    def getLoss(self, lossname):

        if lossname == 'default':
            cross_entropy_loss= nn.CrossEntropyLoss()
            trainloss = cross_entropy_loss
            valloss = cross_entropy_loss
        else:
            # define other losses as and when needed
            pass

        return trainloss, valloss

    def getDataGeneratorObj(self, datageneratortype = 'Default'):
        if datageneratortype == 'Default':
            return DefaultDataGenerator
        else:
            return DefaultDataGenerator

    def train(self, logfilename, pathDirData, pathFileTrain, pathFileVal, modeloutfile, trainerParams):
        self.tb_writer = SummaryWriter(logfilename)
        arch = trainerParams['arch']
        num_epochs = trainerParams['num_epochs']
        optimizername = trainerParams['optimizername']
        lossname = trainerParams['lossname']
        trMaxEpoch = trainerParams['num_epochs']
        num_classes = trainerParams['num_classes']
        batch_size = trainerParams['batch_size']
        checkpoint = trainerParams['checkpoint']
        datageneratortype = trainerParams['datageneratortype']
        first_index = trainerParams['first_index']
        is_trained = trainerParams['useimagenetweights']
        out_activation = trainerParams['out_activation']
        cuda = trainerParams['cuda']
        drop_rate = trainerParams['dropout']
        scheduler_type = trainerParams['scheduler']
        device = torch.device('cuda') if cuda else torch.device('cpu')
        model = self.getModel(trainerParams, device)
        trainloss, valloss = self.getLoss(lossname)
        optimizer = self.getOptimizer(model, optimizername)
        scheduler = self.getScheduler(optimizer = optimizer, scheduler_type = scheduler_type)
        t_0 = trainerParams['t_0'] if scheduler_type == 'cosine_anneal' else num_epochs

        init_lr  = optimizer.state_dict()['param_groups'][-1]['lr']
        if cuda:
            model = torch.nn.DataParallel(model).to(device)

        dataLoaderTrain = self.initDatasetGenerator(datageneratortype = datageneratortype, pathDirData = pathDirData,
                                                    pathFile = pathFileTrain, num_classes = num_classes, batch_size = batch_size,
                                                    first_index = first_index, is_trained = is_trained)
        dataLoaderVal = self.initDatasetGenerator(datageneratortype = datageneratortype, pathDirData = pathDirData,
                                                    pathFile = pathFileVal, num_classes = num_classes, batch_size = batch_size,
                                                    first_index = first_index, is_trained = is_trained)


        lossMin = 100000
        epochID = 0
        restart_epoch = t_0

        for epochID in range(1,num_epochs+1):
            lossValTrain = self.epochTrain(epochID, model, dataLoaderTrain, optimizer, trMaxEpoch, trainloss,
                                           out_activation, device, arch)
            lossVal = self.epochTrain(epochID, model, dataLoaderVal, optimizer, trMaxEpoch, valloss,
                                       out_activation, device, arch)

            scheduler.step(lossVal)
            if lossVal< lossMin:
                lossMIN = lossVal
                torch.save({'epoch':epochID+1, 'state_dict':model.state_dict(), 'best_loss':lossMIN,'optimizer':
                            optimizer.state_dict()}, modeloutfile)

                print('for Epoch', epochID+1, 'file saved with loss', lossVal)
            else:
                print('for Epoch', epochID + 1, 'file saved with loss', lossVal)

        pass

    def getModelOutput(self, model, inputs, out_activation, device, istest):
        varInput = torch.autograd.Variable(inputs).to(device)
        #pdb.set_trace()
        output = model(varInput)
        output = torch.sigmoid(output)

        return output

    def initDatasetGenerator(self, datageneratortype, pathDirData, pathFile, num_classes, batch_size,
                             first_index = 1, shuffle = True, num_workers = 6, is_trained = False):
        generatorObj = self.getDataGeneratorObj()
        dataset = generatorObj(pathImageDirectory = pathDirData, pathDatasetFile = pathFile)
        return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle,
                          num_workers=num_workers, drop_last=True, pin_memory=True)


    def epochTrain(self, epochID, model, dataLoader, optimizer, epochMax, loss, out_activation, device, name):
        model.train()
        lossValTot = 0
        lossValNorm = 0
        losses = AverageMeter()
        pbar = tqdm(dataLoader)
        for batchID,(input, target) in enumerate(pbar):
            target = target.to(device)
            varTarget = torch.autograd.Variable(target).to(device)
            varOutput = self.getModelOutput(model = model, inputs = input, out_activation = out_activation,
                                               istest=False, device = device)
            #varOutput = varOutput.T
            varTarget = np.reshape(varTarget, (8)).type(torch.LongTensor)
            #pdb.set_trace()

            lossvalue = loss(varOutput.to(device), varTarget)
            losses.update(lossvalue.item(), input.size(0))
            lossValTot+=lossvalue.item()
            lossValNorm+=1
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            pbar.set_description("EPOCH[{0}][{1}/{2}]".format(epochID, batchID, len(dataLoader)))
            pbar.set_postfix(loss = "{loss.val:.4f}({loss.avg:.4f})".format(loss = losses))

        outLoss = lossValTot/lossValNorm
        return outLoss

    def epochVal(self, epochID, model, dataLoader, optimizer, epochMax, loss, out_activation, device, name):
        model.eval()
        lossVal = 0
        lossValNorm = 0
        losses = AverageMeter()
        pbar = tqdm(dataLoader)
        for i, (input, target) in enumerate(pbar):
            target = target.to(device)
            with torch.no_grad():
                varTarget = torch.autograd.Variable(target).to(device)
                varOutput = self.getModelOutput(model=model, inputs=input, out_activation=out_activation,
                                                   istest=False, device=device)
                losstensor = loss(varOutput.to(device), varTarget)
                losses.update(losstensor.item(), input.size(0))
                lossVal+=losstensor.item()
                lossValNorm+=1
                pbar.set_description("VALIDATION[{}/{}]".format(i, len(dataLoader)))
                pbar.set_postfix(loss="{loss.val:.4f}({loss.avg:.4f})".format(loss=losses))

        outLoss = lossVal / lossValNorm
        return outLoss