#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:49:53 2018

@author: ashvinsrinivasan
"""
# For control flow process ####mp.save
import warnings
warnings.filterwarnings("ignore")
from dataCheck import FileLoad, DataQualityCheck, DataPreparation
from classificationModel import Logistic, SVM, RandomForestClf, NeuralNetwork
from visualization import Visualisation
from regressionModel import Linear, LassoRegression, RidgeRegression, RandomForestReg
from unsupervised import KMeansclustering
from saveResults import Results
from saveResults import plots, regressionPlots, classificationPlots
import SentimentAnalysis as sa
import cnnClassification as c
import pandas as pd
import os
import matplotlib.pyplot as plt
class WorkFlow:
    def __init__(self, configDf = None):
        #self.configFilepath=configFilepath
        #configDf=pd.read_csv(configFilepath, sep = ',')
        #configDf.set_index('Parameters',inplace = True)
        columnsConsidered=configDf.loc['ColumnsConsidered','Value']
        imputation = configDf.loc['Imputation','Value']
        target= configDf.loc['Target','Value']
        clf=configDf.loc['Classifier','Value']
        modelingType = configDf.loc['ModelingType','Value']
        filepath = configDf.loc['filepath','Value']
        separator = configDf.loc['separator','Value']
        temp1=DataQualityCheck(filepath,target = target, separator = separator, columnsConsidered=columnsConsidered)
        df1=temp1.considerColumns()
        self.dfCopy = df1.copy(deep = True)
        temp1.checkMissingValues()
        temp1.checkOutliers()
        plot = plots()
        plot.corr_plot(df1, clf)
        plot.hist(df1)
        plot.savePairPlots(df1)
        plot.saveBoxPlots(df1)


        temp2=DataPreparation(imputation=imputation, target = target )
        df1=temp2.convertCategoricalToDummy(df1)
        df1=temp2.imputation(df1)
        df1=temp2.featureNormalisation(df1)

        if modelingType == 'classification':
            if  clf == 'Logistic':
                temp3 = Logistic(target = target, df = df1)
                scores = temp3.runLogistic()
                ypred, y_test, ypredProb = temp3.predLogistic()
                #self.callClassificationPlots( scores, clf, y_test, ypred, df1[target],ypredProb)
            elif clf == 'SVM':
                temp3 = SVM(target = target, df = df1)
                scores,optparams = temp3.runSVM()
                ypred, y_test, ypredProb = temp3.predSVM(optparams)
                #self.callClassificationPlots(scores, clf, y_test, ypred, df1[target])
            elif clf == 'RF':
                temp3 = RandomForestClf(target = target, df = df1)
                scores, optparams = temp3.runRF()
                ypred, y_test, ypredProb = temp3.predRF(optparams)
                #self.callClassificationPlots(scores, clf, y_test, ypred, df1[target])
            elif clf == 'NN':
                temp3 = NeuralNetwork(target = target, df = df1)
                scores, model = temp3.runNN()
                ypred, y_test, ypredProb = temp3.predNN()

            self.callClassificationPlots( scores, clf, y_test, ypred, df1[target],ypredProb)


        elif modelingType == 'regression':
            if clf == 'Linear':
                temp3 = Linear(target = target, df = df1)
                scores = temp3.runLinear()
                ypred, y_test = temp3.predLinear()
            elif clf == 'LassoRegression':
                temp3 = LassoRegression(target = target, df = df1)
                scores, optparams = temp3.runLasso()
                ypred, y_test = temp3.predLasso(optparams)
            elif clf == 'RidgeRegression':
                temp3 = RidgeRegression(target = target, df = df1)
                scores, optparams = temp3.runRidge()
                ypred, y_test = temp3.predRidge(optparams)

            elif clf =='RandomForest':
                temp3 = RandomForestReg(target = target, df = df1)
                scores, optparams = temp3.runRF()
                ypred, y_test = temp3.predRF(optparams)

            self.callRegressionPlots(df1, target, ypred, y_test, clf)



        #print('ypred: {}'.format(ypred))
        print('scores for classifier {} are: {}'.format(configDf.loc['Classifier','Value'],scores))

    def callClassificationPlots(self, scores, clf, y_test, ypred, targetSeries, ypredProbaility):
            Results(scores, clf)
            plotimg = plots()
            plotimg.conf_matrix(y_test, ypred, clf, targetSeries.unique())

            clfPlots = classificationPlots()
            clfPlots.ROCplots(y_test,ypredProbaility,clf)
            clfPlots.precisionVsrecallPlots(y_test,ypredProbaility,clf)



    def callRegressionPlots(self, df, target, ypred, ytest, clf):
            regplots = regressionPlots()
            regplots.jointPlot(df, target)
            regplots.swarmPlots(self.dfCopy, target)
            regplots.normalQQ(ypred, ytest, clf)
            regplots.residVsFitted(ypred, ytest, clf)



class SentimentWorkflow(Visualisation):
    def __init__(self, df = None):
        #configFilepath = r"C:\Users\ashvin\Desktop\UnderDevelopment\sentimentConfig.csv"
        #df = pd.read_csv(configFilepath)
        #df.set_index('Parameters', inplace = True)
        print('creating Results directory--------> at {}'.format(os.getcwd()))
        try:
            os.makedirs('results')
        except:
            pass
        reviews = df.loc['paths','Value'].split(',')
        catNum = df.loc['numOfCat','Value']
        catTypes = df.loc['catType', 'Value'].split(',')
        sentimentLabel = df.loc['train','Value']
        sentiment = sa.SentimentTrain()
        acc, allReviews = sentiment.extract(reviews, catNum, catTypes, sentimentLabel)
        visuals=Visualisation(df = None, target = None)
        visuals.wordCloud(allReviews)



class UnsupervisedWorkflow:
       def __init__(self, configDf = None):
           #configDf = pd.read_csv(configFilepath)
           #configDf.set_index('Parameters', inplace = True)
           filepath = configDf.loc['filepath','Value']
           data = pd.read_csv(filepath)
           kmeanscluster = KMeansclustering(data, clusterSizes = None)

class CV:
    def __init__(self, configDf = None):
        if configDf.loc['demo', 'Value'] == 'True':
            batch_size = configDf.loc['batch_size', 'Value']
            num_classes = configDf.loc['num_classes', 'Value']
            epochs = configDf.loc['Epochs', 'Value']
            opt = configDf.loc['optimizer', 'Value']
            data_augment = configDf.loc['data_augmentation', 'Value']
            cnnNet = c.cnn(batch_size,num_classes,epochs,opt, data_augment)
            cnnNet.demo()
