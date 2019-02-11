import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from visualization import Visualisation

class Results:
    def __init__(self, acc, clfName = None, params = None):
        try:
            os.mkdir('results')

        except:
            pass



        resArray = acc
        metricList = ['acc']
        saveDict= {}
        for metric in metricList:
            try:
                saveDict[metric] = resArray.tolist()
            except:
                pass
        saveDict = json.dumps(saveDict)
        fileName = 'data_{}.json'.format(clfName)
        path = os.path.join(os.getcwd(),'results',fileName)
        with open(path, 'w') as outfile:
            json.dump(saveDict, outfile)


class plots:
    def __init__(self):
        try:
            os.mkdir('results')

        except:
            pass
    def conf_matrix(self,y_test, y_pred, clf, uniqueNames):
        cnf_matrix = confusion_matrix(y_test.values, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        vis = Visualisation()
        class_names = uniqueNames
        vis.plot_confusion_matrix( clf, cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
    def corr_plot(self, df, clf):
            vis = Visualisation()
            vis.correlationPlot(df,clf)

    def hist(self, df):
        vis = Visualisation()
        vis.histogram(df)

    def savePairPlots(self, df):
        vis = Visualisation()
        vis.pairplot(df)

    def saveBoxPlots(self, df):
        vis = Visualisation()
        vis.boxplot(df)



class regressionPlots:
    def __init__(self):
        pass

    def jointPlot(self, df, target):
            vis = Visualisation()
            vis.snsJointPlots(df, target)
    def swarmPlots(self, df, target):
            vis = Visualisation()
            vis.snsSwarmPlots(df, target)
    def normalQQ(self, ypred, ytest, clf):
            vis = Visualisation()
            vis.normalQQplots(ypred, ytest, clf)
    def residVsFitted(self, ypred, ytest, clf):
            vis = Visualisation()
            vis.residVsPredPlots(ypred, ytest, clf)

class classificationPlots:
    def __init__(self):
        pass

    def ROCplots(self, ytest, ypred_prob, clf):
                    vis = Visualisation()
                    vis.ROCplots(ytest, ypred_prob, clf)

    def precisionVsrecallPlots(self, ytest, ypred_prob, clf):
                    vis = Visualisation()
                    vis.precVsRecall(ytest, ypred_prob, clf)
