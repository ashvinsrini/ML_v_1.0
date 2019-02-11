import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.regressor import ResidualsPlot
class RegressionModel:

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

    def crossValidation(self,regr,X_train,y_train):

        scores = cross_val_score(regr, X_train, y_train, cv=5, scoring='r2')
        return scores

    def buildModel(self, mod, X_train, y_train, X_test, y_test, optparams = None):
            mod.fit(X_train, y_train)
            ypred = mod.predict(X_test)
            print('ypred is {}'.format(ypred))
            print('ytest is {}'.format(y_test))
            return ypred,y_test

class Linear(RegressionModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df

        self.regr = LinearRegression()
        RegressionModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = RegressionModel.dataSplit(self)
    def runLinear(self):
        return RegressionModel.crossValidation(self, self.regr, X_train=self.X_train, y_train=self.y_train)

    def predLinear(self):
            return RegressionModel.buildModel(self, self.regr, self.X_train,self.y_train,self.X_test,self.y_test)


class LassoRegression(RegressionModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df

        #self.regr = Lasso()
        RegressionModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = RegressionModel.dataSplit(self)
    def runLasso(self):

        np = self.np
        params={}
        params['alpha'] = [2**-5, 2**-4, 2**-3,2**-1,2**0,2**2,2**4, 2**5, 2**6]
        scores = []
        for alpha in params['alpha']:
            self.regr = Lasso(alpha = alpha)
            scores.append(np.mean(RegressionModel.crossValidation(self, regr = self.regr, X_train=self.X_train, y_train=self.y_train)))
        alphaOptimized = np.where(scores == np.max(scores))
        optParams = {}
        optParams['alpha'] = params['alpha'][alphaOptimized[0][0]]
        print('Best alpha is {}'.format(params['alpha'][alphaOptimized[0][0]]))
        #print(alphaOptimized[0][0])

        return scores,optParams
    def predLasso(self, optParams):
            optRegr = Lasso(alpha = optParams['alpha'])
            return RegressionModel.buildModel(self, optRegr, self.X_train,self.y_train,self.X_test,self.y_test)





class RidgeRegression(RegressionModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df

        #self.regr = Lasso()
        RegressionModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = RegressionModel.dataSplit(self)
    def runRidge(self):

        np = self.np
        params={}
        params['alpha'] = [2**-5, 2**-4, 2**-3,2**-1,2**0,2**2,2**4, 2**5, 2**6]
        scores = []
        for alpha in params['alpha']:
            self.regr = Ridge(alpha = alpha)
            scores.append(np.mean(RegressionModel.crossValidation(self, regr = self.regr, X_train=self.X_train, y_train=self.y_train)))
        alphaOptimized = np.where(scores == np.max(scores))
        optParams = {}
        optParams['alpha'] = params['alpha'][alphaOptimized[0][0]]
        print('Best alpha is {}'.format(params['alpha'][alphaOptimized[0][0]]))
        #print(alphaOptimized[0][0])

        return scores,optParams
    def predRidge(self, optParams):
            optRegr = Ridge(alpha = optParams['alpha'])
            return RegressionModel.buildModel(self, optRegr, self.X_train,self.y_train,self.X_test,self.y_test)


class RandomForestReg(RegressionModel):
    def __init__(self, target= None, df = None):
        self.target = target
        self.df = df

        #self.regr = Lasso()
        RegressionModel.__init__(self, target = self.target, df = self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = RegressionModel.dataSplit(self)
    def runRF(self):

        np = self.np
        params = {}
        params['n_estimators']=[5,10,15]
        params['min_samples_split']=[2,10,50]
        scores=[]
        for estimator in params['n_estimators']:
            for sample in params['min_samples_split']:
                  self.regr = RandomForestRegressor(n_estimators = estimator, min_samples_split = sample)
                  scores.append(np.mean(RegressionModel.crossValidation(self, regr = self.regr, X_train=self.X_train, y_train=self.y_train)))
        scores = np.reshape(scores, (len(params['n_estimators']), len(params['min_samples_split'])))
        estimatorMax, samplesMax = np.where(scores == np.max(scores))
        optParams = {}
        optParams['n_estimators'] = params['n_estimators'][estimatorMax[0]]
        optParams['min_samples_split'] = params['min_samples_split'][samplesMax[0]]
        print('Best estimator value {}, min_samples_split {}'.format(params['n_estimators'][estimatorMax[0]], params['min_samples_split'][samplesMax[0]]))
        return scores, optParams

    def predRF(self, optParams):
            optRegr = RandomForestRegressor(n_estimators = optParams['n_estimators'],
                                  min_samples_split = optParams['min_samples_split'])
            return RegressionModel.buildModel(self, optRegr, self.X_train,self.y_train,self.X_test,self.y_test)
