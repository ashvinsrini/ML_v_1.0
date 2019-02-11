#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:49:53 2018

@author: ashvinsrinivasan
"""
# For control flow process ####mp.save
from dataCheck import FileLoad, DataQualityCheck, DataPreparation
from classificationModel import Logistic, SVM, RandomForestClf, NeuralNetwork
from visualization import Visualisation
from regressionModel import Linear, LassoRegression, RidgeRegression, RandomForestReg
from unsupervised import KMeansclustering
import SentimentAnalysis as sa
import pandas as pd
class WorkFlow: 
    def __init__(self, configFilepath = None):
        self.configFilepath=configFilepath
        configDf=pd.read_csv(configFilepath, sep = ',')
        configDf.set_index('Parameters',inplace = True)
        columnsConsidered=configDf.loc['ColumnsConsidered','Value']
        imputation = configDf.loc['Imputation','Value']
        target= configDf.loc['Target','Value']
        clf=configDf.loc['Classifier','Value']
        modelingType = configDf.loc['ModelingType','Value']
        filepath = configDf.loc['filepath','Value']
        separator = configDf.loc['separator','Value']
        temp1=DataQualityCheck(filepath,target = target, separator = separator, columnsConsidered=columnsConsidered)        
        df1=temp1.considerColumns()
        temp1.checkMissingValues()
        temp1.checkOutliers()
        
        
        temp2=DataPreparation(imputation=imputation )
        df1=temp2.convertCategoricalToDummy(df1)
        df1=temp2.imputation(df1)
        df1=temp2.featureNormalisation(df1)
        
        if modelingType == 'classification':
            if  clf == 'Logistic':
                temp3 = Logistic(target = target, df = df1)
                scores = temp3.runLogistic()
            elif clf == 'SVM':
                temp3 = SVM(target = target, df = df1)
                scores = temp3.runSVM()
            elif clf == 'RF':
                temp3 = RandomForestClf(target = target, df = df1)
                scores = temp3.runRF()
            elif clf == 'NN':   
                temp3 = NeuralNetwork(target = target, df = df1)
                scores = temp3.runNN()
                
        elif modelingType == 'regression': 
            if clf == 'Linear':
                temp3 = Linear(target = target, df = df1)
                scores = temp3.runLinear()
            elif clf == 'LassoRegression':
                temp3 = LassoRegression(target = target, df = df1)
                scores = temp3.runLasso()
            elif clf == 'RidgeRegression':
                temp3 = RidgeRegression(target = target, df = df1)
                scores = temp3.runRidge()  
            elif clf =='RandomForest':
                temp3 = RandomForestReg(target = target, df = df1)
                scores = temp3.runRF()  
                
                
            
            
        print('scores for classifier {} are: {}'.format(configDf.loc['Classifier','Value'],scores))
        
        temp4=Visualisation(df = df1, target = target)
        #temp4.plot()        
        #temp4.correlationPlot()

class SentimentWorkflow(Visualisation):
    def __init__(self,configFilepath = None):
        #configFilepath = r"C:\Users\ashvin\Desktop\UnderDevelopment\sentimentConfig.csv"
        df = pd.read_csv(configFilepath)  
        df.set_index('Parameters', inplace = True)
        reviews = df.loc['paths','Value'].split(',')
        catNum = df.loc['numOfCat','Value']
        catTypes = df.loc['catType', 'Value'].split(',')
        sentimentLabel = df.loc['train','Value']
        sentiment = sa.SentimentTrain()
        acc, allReviews = sentiment.extract(reviews, catNum, catTypes, sentimentLabel)
        visuals=Visualisation(df = None, target = None)
        visuals.wordCloud(allReviews)
        
    

class UnsupervisedWorkflow:
       def __init__(self, configFilepath = None):
           configDf = pd.read_csv(configFilepath)  
           configDf.set_index('Parameters', inplace = True)
           filepath = configDf.loc['filepath','Value']
           data = pd.read_csv(filepath)
           kmeanscluster = KMeansclustering(data, clusterSizes = None)
           
           
    
    

