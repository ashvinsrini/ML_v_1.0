# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time
import sys
import pickle
from Trainer import *
from DataGenerator import *
os.environ["PYTHONHTTPSVERIFY"] = "0"
def main ():
    runTrain()

def runTrain():
    code_dir = '/Users/ashvinsrinivasan/Desktop/'
    data_dir = os.path.join(code_dir, 'MNIST')
    data_subdir = os.path.join(data_dir, 'trial1')
    runs_dir = os.path.join(data_dir, 'runs')
    params = TrainerParams(
                    nnArchitecture = 'Default',
                    nnIsTrained = True,
                     nnClassCount = 10,
                    trBatchSize = 8,
                     trMaxEpoch = 10,
                    scheduler = 'lrplateau',
                    lossname = 'default',
                    evalloss = 'default',
                     datageneratortype = 'Default',
                     optimizername= 'Adam',
                     cuda = False,
                     t_0 = 1000,
                     t_mult = 1,
                     first_index = 1,
                    out_activation = None
                            ).getparams()


    pathDirData = ''
    pathFileTrain = os.path.join(data_subdir,'train.csv')
    pathFileVal = os.path.join(data_subdir, 'val.csv')
    pathFileTest = os.path.join(data_subdir, 'test.csv')
    logfilename = 'new_model'
    logfilename = os.path.join(runs_dir, logfilename)
    if params['cuda']:
        cudastr = "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = cudastr
    pathModel = os.path.join(logfilename,'m-'+'.pth.tar')
    TrainerObj = BaseTrainer()
    print('Training NN architecture = ', params['arch'])
    TrainerObj.train(logfilename, pathDirData, pathFileTrain, pathFileVal, pathModel, params)
    modelstatsdir = os.path.join(logfilename, 'model_stats')
    if not os.path.exists(modelstatsdir):
        os.mkdir(modelstatsdir)

    params['data_dir'] = pathDirData
    params['train_file'] = pathFileTrain
    params['val_file'] = pathFileVal
    params['logfilename'] = logfilename
    params['checkpoint'] = pathModel
    print('Testing the trained model')
    params['batch_size'] = 1
    pickle.dump(params, open(os.path.join(logfilename,'train_info_.p'),'wb'))
    #TrainerObj.test
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    #print(a)
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
