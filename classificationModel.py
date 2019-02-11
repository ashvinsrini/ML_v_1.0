import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier


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
    
    
class Logistic(ClassificationModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df
        self.clf = LogisticRegression()
        ClassificationModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
    def runLogistic(self):    
        return ClassificationModel.crossValidation(self, clf = self.clf, X_train=self.X_train, y_train=self.y_train)
        
class SVM(ClassificationModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df
        ClassificationModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
    def runSVM(self):    
        
        
        #self.clf = svm.SVC()
        params = {}
        params['C']= [2**-3,2**-1,2**0,2**2,2**4,2**6,2**8,2**10]
        params['kernel'] =['linear','rbf']
        np = self.np
        scores=[]
        for cost in params['C']:
            for kernel in params['kernel']:
               self.clf = svm.SVC(C = cost, kernel = kernel) 
               scores.append(np.mean(ClassificationModel.crossValidation(self, clf = self.clf, X_train=self.X_train, y_train=self.y_train)))
        scores = np.reshape(scores, (len(params['C']), len(params['kernel'])))
        CMax, kernelMax = np.where(scores == np.max(scores))
        print('Best C value {}, kernel {}'.format(params['C'][CMax[0]], params['kernel'][kernelMax[0]]))
        return scores       
        
        
        
                     
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
        print('Best estimator value {}, min_samples_split {}'.format(params['n_estimators'][estimatorMax[0]], params['min_samples_split'][samplesMax[0]]))
        return scores 
    
    
class NeuralNetwork:
        def __init__(self, target = None, df = None):
            self.target = target
            self.df = df
            ClassificationModel.__init__(self, target = self.target, df = self.df)
            self.X_train, self.X_test, self.y_train, self.y_test = ClassificationModel.dataSplit(self)
        def runNN(self,hiddenLayerNeurons = 50, dropOutFirstLayer = 0.3, dropOutSecondLayer = 0.2,noOfEpochs = 100, \
                  batchSize = 32):

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
            return scores
            
        
        
