from django.shortcuts import render
from basicapp import forms, unsupervised, nlp, cv
import pandas as pd
import WorkFlow
import django
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.http import HttpResponse
#from WorkFlow import WorkFlow, SentimentWorkflow, UnsupervisedWorkflow

# Create your views here.

def index(request):
    return render(request, 'basicapp/index.html')

def form_name_view(request):
    form = forms.FormName()
    if request.method == 'POST':
        form =  forms.FormName(request.POST)
    if form.is_valid():
        print('validation success')
        #print("text:"+form.cleaned_data['input'])
        #df = pd.read_csv(form.cleaned_data['input'])
        #df.to_csv(form.cleaned_data['output'])
        entries = ['filepath', 'label', 'ColumnsConsidered','Imputation','Target','Classifier','ModelingType','separator']
        df = pd.DataFrame()
        i = 0
        for entry in entries:
            df.loc[i,'Parameters'] = entry
            df.loc[i,'Value'] = form.cleaned_data[entry]
            i+=1
        #df.to_csv('/home/ashvin/Desktop/todelete/titanic.csv')
        df.set_index('Parameters', inplace = True)
        print(df.loc['label','Value'])
        if df.loc['label','Value'] == 'sentiment':
            WorkFlow.SentimentWorkflow(configDf = df)
        elif df.loc['label','Value'] == 'supervised':
            WorkFlow.WorkFlow(configDf = df)
        elif df.loc['label','Value'] == 'unsupervised':
            WorkFlow.UnsupervisedWorkflow(configDf = df)


    return render(request, 'basicapp/form_page.html',{'form':form})

def unsupervise(request):
    form = unsupervised.FormName()
    if request.method == 'POST':
        form =  unsupervised.FormName(request.POST)
    if form.is_valid():
        print('validation success')
        #print("text:"+form.cleaned_data['input'])
        #df = pd.read_csv(form.cleaned_data['input'])
        #df.to_csv(form.cleaned_data['output'])
        entries = ['filepath', 'label', 'ColumnsConsidered','separator']
        df = pd.DataFrame()
        i = 0
        for entry in entries:
            df.loc[i,'Parameters'] = entry
            df.loc[i,'Value'] = form.cleaned_data[entry]
            i+=1
        #df.to_csv('/home/ashvin/Desktop/todelete/titanic.csv')
        df.set_index('Parameters', inplace = True)
        print(df.loc['label','Value'])
        WorkFlow.UnsupervisedWorkflow(configDf = df)


    return render(request, 'basicapp/unsupervised.html',{'form':form})

def NLP(request):
    form = nlp.FormName()
    if request.method == 'POST':
        form =  nlp.FormName(request.POST)
    if form.is_valid():
        print('validation success')
        #print("text:"+form.cleaned_data['input'])
        #df = pd.read_csv(form.cleaned_data['input'])
        #df.to_csv(form.cleaned_data['output'])
        entries = ['label', 'paths', 'numOfCat','catType','train']
        df = pd.DataFrame()
        i = 0
        for entry in entries:
            df.loc[i,'Parameters'] = entry
            df.loc[i,'Value'] = form.cleaned_data[entry]
            i+=1
        #df.to_csv('/home/ashvin/Desktop/todelete/titanic.csv')
        df.set_index('Parameters', inplace = True)
        print(df.loc['label','Value'])
        WorkFlow.SentimentWorkflow(df)


    return render(request, 'basicapp/nlp.html',{'form':form})

def computervision(request):
    form = cv.FormName()
    if request.method == 'POST':
        form =  cv.FormName(request.POST)
    if form.is_valid():
        print('validation success')
        #print("text:"+form.cleaned_data['input'])
        #df = pd.read_csv(form.cleaned_data['input'])
        #df.to_csv(form.cleaned_data['output'])
        entries = ['batch_size', 'num_classes', 'Epochs','optimizer','data_augmentation','demo',
                    'filepath','target','imght','imgwdt','rgb']
        df = pd.DataFrame()
        i = 0
        for entry in entries:
            df.loc[i,'Parameters'] = entry
            df.loc[i,'Value'] = form.cleaned_data[entry]
            i+=1
        #df.to_csv('/home/ashvin/Desktop/todelete/titanic.csv')
        df.set_index('Parameters', inplace = True)
        WorkFlow.CV(configDf = df)



    return render(request, 'basicapp/cv.html',{'form':form})



def plots(request):
    f = matplotlib.figure.Figure()
    FigureCanvas(f)
    ax = f.add_subplot(111)
    x = np.arange(-2,1.5,.01)
    y = np.sin(np.exp(2*x))
    ax.plot(x, y)
    ax.autoscale()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(f)
    response = HttpResponse(buf.getvalue(), content_type='image/png')

    return response
