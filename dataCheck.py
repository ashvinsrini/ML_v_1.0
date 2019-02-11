import pandas as pd
class FileLoad:
    def __init__(self, filepath, separator):
        #self.filepath=input('please enter csv filepath\n')
        self.filepath = filepath
        self.df=pd.read_csv(self.filepath, sep = separator)
        

class DataQualityCheck(FileLoad):
    
    def __init__(self, filepath, target, separator, columnsConsidered=None):
        #d.load()
        FileLoad.__init__(self, filepath, separator)
        #self.col_names=input('please enter the coulmns to be deleted paranthised by []\n')
        self.colNames = columnsConsidered
        self.target = target
    
    def considerColumns(self):
        #loader=file_load()
        #filepath=self.filepath
        if self.colNames == 'None':
            print('if condition checked properly')
            colNames = list(self.df.columns)
            
            
        else :
            colNames = self.colNames.split(',')
        
        self.df = self.df[colNames]
        #df.drop(colNames,axis=1,inplace=True)
        print(colNames)
        return self.df
    
    def dropNA(self):
        df = self.df
        df = df.dropna(inplace = True)
        return df
        
        
    def checkMissingValues(self):
        df = self.df 
        self.names = [i for i in df.columns if df[i].dtype.name == 'float64']
        names=self.names
        ind = {}
        for name in names: 
            ind[name] = df[name].isnull() 
            print('{} has {} missing entries'.format(name, sum(ind[name])))
        pass   

    def checkOutliers(self):
        print('yet to implement modified z score for outlier treatment')
    
    
    def saveDf(self,df):
        import pandas as pd
        self.outputPath=input('enter output path\n')
        outputPath=self.outputPath
        df.to_csv(outputPath)
        

class DataPreparation:
    
    def __init__(self, imputation='mean', outlier = 'mean'):
        #self.type=input('please enter the type of imputation\n')
        self.type = imputation
        print(self.type)
        
    def convertCategoricalToDummy(self,df):
        import pandas as pd
        df=pd.get_dummies(df)
        return df        
        
    
    def imputation(self, df):
        #df.drop('Unnamed: 0',inplace=True,axis=1)
        type=self.type
        self.names = [i for i in df.columns if df[i].dtype.name == 'float64']
        names=self.names
        if type=='mean':        
            for name in names:
                df[name].fillna(df[name].mean(),inplace=True)
        elif type == 'mode':
            for name in names:
                df[name].fillna(df[name].mode(),inplace=True)
        else :
            for name in names:
                df[name].fillna(df[name].median(),inplace=True)                
                
        return df       

    def outlierImputation(self, df):
        type=self.type
        self.names = [i for i in df.columns if df[i].dtype.name == 'float64']
        names=self.names
        
        
        
        
        
        
         
    
    def featureNormalisation(self,df):
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        for name in self.names:
            df[name]=scaler.fit_transform(df[[name]])
        return df
        

                
        
        
    


#temp=data_quality_check()        
#df1=temp.drop_columns()
