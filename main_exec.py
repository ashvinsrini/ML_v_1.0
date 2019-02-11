#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:51:01 2018

@author: ashvinsrinivasan
"""


from WorkFlow import WorkFlow, SentimentWorkflow, UnsupervisedWorkflow
import pandas as pd
def main():
    configFilepath=r"/Users/ashvinsrinivasan/Desktop/UnderDevelopment/test_cases/iris_config.csv"
    df = pd.read_csv(configFilepath, sep = ',')
    
    df.set_index('Parameters', inplace = True)
    if df.loc['label','Value'] == 'sentiment':
        SentimentWorkflow(configFilepath)
    elif df.loc['label','Value'] == 'supervised':
        WorkFlow(configFilepath)
    elif df.loc['label','Value'] == 'unsupervised':
        UnsupervisedWorkflow(configFilepath)
        
if __name__ == '__main__':
    main()
