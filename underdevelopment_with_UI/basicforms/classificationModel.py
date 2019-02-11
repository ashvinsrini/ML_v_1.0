import warnings
warnings.filterwarnings("ignore")
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from saveResults import plots
from saveResults import Results
#from optimalModellingClasses import


class ClassificationModel:

    def __init__(self, target=None, df = None):
        #self.target=input('Enter the target variable\n')
        #self.clf=input('Enter the type of classifier\n')
        #print('1) Logistic Classifier \n 2) SVM\n')

        self.np = np
        self.target=target

    def dataSplit(self):
        df=self.df
        self.headers=df.columns
        target=self.target
        headers=self.headers
        y=df[target]
        X=df[headers[headers!=target]]
        X_train,X_test,y_train,y_test=train_test_split(X,y)
        params=[X_train,X_test,y_train,y_test]
        return params


    def crossValidation(self,clf,X_train,y_train):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        return scores

    def buildModel(self, mod, X_train, y_train, X_test, y_test, optparams = None):
            mod.fit(X_train, y_train)
            ypred = mod.predict(X_test)
            ypredProb = mod.predict_proba(X_test)[:,1]
            return ypred,y_test, ypredProb





class Logistic(ClassificationModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df
        self.clf = LogisticRegression()
        ClassificationModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
    def runLogistic(self):
        return ClassificationModel.crossValidation(self, clf = self.clf, X_train=self.X_train, y_train=self.y_train)
    def predLogistic(self):
            return ClassificationModel.buildModel(self, self.clf, self.X_train,self.y_train,self.X_test,self.y_test)

class SVM(ClassificationModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df
        ClassificationModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
    def runSVM(self):


        #self.clf = svm.SVC()
        params = {}
        params['C']= [2**0]
        params['kernel'] =['rbf']
        np = self.np
        scores=[]
        for cost in params['C']:
            for kernel in params['kernel']:
               self.clf = svm.SVC(C = cost, kernel = kernel, probability = True)
               scores.append(np.mean(ClassificationModel.crossValidation(self, clf = self.clf, X_train=self.X_train, y_train=self.y_train)))
        scores = np.reshape(scores, (len(params['C']), len(params['kernel'])))
        CMax, kernelMax = np.where(scores == np.max(scores))
        optParams = {}
        optParams['C'] = params['C'][CMax[0]]
        optParams['kernel'] = params['kernel'][kernelMax[0]]
        print('Best C value {}, kernel {}'.format(params['C'][CMax[0]], params['kernel'][kernelMax[0]]))
        return scores,optParams
    def predSVM(self, optparams):
                optClf = svm.SVC(C  = optparams['C'],
                                                    kernel = optparams['kernel'], probability = True)
                return ClassificationModel.buildModel(self, optClf, self.X_train,self.y_train,self.X_test,self.y_test)





class RandomForestClf(ClassificationModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df

        ClassificationModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
    def runRF(self):

        #self.clf=  RandomForestClassifier()
        params = {}
        params['n_estimators']=[5,10,15]
        params['min_samples_split']=[2,10,50]
        np = self.np
        scores=[]
        for estimator in params['n_estimators']:
            for sample in params['min_samples_split']:
                  self.clf = RandomForestClassifier(n_estimators = estimator, min_samples_split = sample)
                  scores.append(np.mean(ClassificationModel.crossValidation(self, clf = self.clf, X_train=self.X_train, y_train=self.y_train)))
        scores = np.reshape(scores, (len(params['n_estimators']), len(params['min_samples_split'])))
        estimatorMax, samplesMax = np.where(scores == np.max(scores))
        optParams = {}
        optParams['n_estimators'] = params['n_estimators'][estimatorMax[0]]
        optParams['min_samples_split'] = params['min_samples_split'][samplesMax[0]]

        print('Best estimator value {}, min_samples_split {}'.format(params['n_estimators'][estimatorMax[0]], params['min_samples_split'][samplesMax[0]]))

        return scores, optParams
    def predRF(self, optparams):
                print(optparams)
                optClf = RandomForestClassifier(n_estimators = optparams['n_estimators'],
                                                min_samples_split = optparams['min_samples_split'])
                return ClassificationModel.buildModel(self, optClf, self.X_train,self.y_train,self.X_test,self.y_test)


class NeuralNetwork:
        def __init__(self, target = None, df = None):
            self.target = target
            self.df = df
            ClassificationModel.__init__(self, target = self.target, df = self.df)
            self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
        def runNN(self,hiddenLayerNeurons = 50, dropOutFirstLayer = 0.3, dropOutSecondLayer = 0.2,noOfEpochs = 100, \
                  batchSize = 32):
            if batchSize == None:
                batchSize = len(self.X_train)
            _,c = self.X_train.shape

            model = models.Sequential()
            # Input - Layer
            model.add(layers.Dense(hiddenLayerNeurons, activation = "relu", input_shape=(c, )))
            # Hidden - Layers
            model.add(layers.Dropout(dropOutFirstLayer, noise_shape=None, seed=None))
            model.add(layers.Dense(hiddenLayerNeurons, activation = "relu"))
            model.add(layers.Dropout(dropOutSecondLayer, noise_shape=None, seed=None))
            model.add(layers.Dense(hiddenLayerNeurons, activation = "relu"))
            # Output- Layer
            model.add(layers.Dense(1, activation = "sigmoid"))
            model.summary()
            # compiling the model
            model.compile(
             optimizer = "adam",
             loss = "binary_crossentropy",
             metrics = ["accuracy"]
            )
            results = model.fit(
             self.X_train, self.y_train,
             epochs= noOfEpochs,
             batch_size = batchSize,
             validation_data = (self.X_test, self.y_test)
            )
            scores = np.mean(results.history["val_acc"])

            #Results(scores, clfName = 'NN')
            #plotimg = plots()
            #plotimg.conf_matrix(self.y_test, ypred, clf = 'NN')
            self.model = model
            return scores, model
        def predNN(self, optparams = None):
                ypred = self.model.predict_classes(self.X_test)
                ypred_prob = self.model.predict(self.X_test)
                return ypred, self.y_test, ypred_prob
