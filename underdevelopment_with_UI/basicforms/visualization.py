import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import itertools
import statsmodels.api as sm
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
import pickle
class Visualisation():
        def __init__(self, df = None ,target = None):
            #file_load.__init__(self)
            self.target = target
            self.df = df

        def plot(self):
            df = self.df
            plt.figure()
            target = self.target

            plt.hist(df[target])
            plt.show()

        def histogram(self, df):
               try:
                           os.makedirs(os.path.join('results', 'histogram'))
               except:
                           pass
               for name in df.columns.values:

                        try:

                           plt.figure()
                           plt.title('histogram: {}'.format(name))
                           sns.distplot(df[name])
                           fileName = name+'histogram'+'.png'
                           path = os.path.join(os.getcwd(),'results/histogram',fileName)
                           plt.savefig(path)
                        except:
                           pass

        def boxplot(self,df):
                    try:
                        os.makedirs(os.path.join('results', 'boxplots'))
                    except:
                        pass
                    for name in df.columns.values:
                                try:
                                    plt.figure()
                                    plt.title('boxplot: {}'.format(name))
                                    sns.boxplot(df[name])
                                    fileName = name+'boxplot'+'.png'
                                    path = os.path.join(os.getcwd(),'results/boxplots',fileName)
                                    plt.savefig(path)
                                except:
                                    pass


        def pairplot(self, df):
                    sns_pairplot = sns.pairplot(df, diag_kind="kde")
                    fileName = 'pairplot'+'.png'
                    path = os.path.join(os.getcwd(),'results',fileName)
                    plt.savefig(path)




        def correlationPlot(self, df ,clf):
            #df = self.df

            #self.names = [i for i in df.columns if df[i].dtype.name == 'float64']
            corr = df.corr()

            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            fileName = clf+'correlation'+'.png'
            path = os.path.join(os.getcwd(),'results',fileName)
            plt.savefig(path)

        def wordCloud(self, allReviews):
            i = 1
            for cat in allReviews.keys():
                wordcloud = WordCloud().generate(allReviews[cat])
                plt.figure(i)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title('{}'.format(cat))
                fileName = '{}.png'.format(cat)
                path = os.path.join(os.getcwd(), 'results',fileName)
                plt.savefig(path)
                i+= 1


        def silhouetteScores(self, scores, clusterSizes):
            try:
                           os.makedirs('results')
            except:
                           pass
            plt.figure()
            plt.plot(clusterSizes,scores)
            plt.xlabel('cluster size')
            plt.ylabel('average silhouette scores')
            plt.title('silhouette scores vs cluster size')
            fileName = 'silhouette_scores.png'
            path = os.path.join(os.getcwd(),'results',fileName)
            plt.savefig(path)

        def snsJointPlots(self, df, target = None):
               try:
                    os.makedirs(os.path.join('results', 'JointPlots'))
               except:
                    pass
               names = list(df.columns.values)
               names.remove(target)
               #print('target----------->{}'.format(target))
               for name in names:
                        try:
                            plt.figure()
                            plt.title('Jointplot: {}'.format(name))
                            sns.jointplot(x=name, y=target, data=df, kind="reg");
                            fileName = name+'JointPlots'+'.png'
                            path = os.path.join(os.getcwd(),'results/JointPlots',fileName)
                            plt.savefig(path)
                        except:
                            pass

        def snsSwarmPlots(self, df, target):
               try:
                    os.makedirs(os.path.join('results', 'SwarmPlots'))
               except:
                    pass

               try:

                     names = list(df.select_dtypes(include=['category','object']).columns.values)
               #names.remove(target)


               #print('target----------->{}'.format(target))
                     for name in names:
                                 plt.figure()
                                 plt.title('Swarmplot: {}'.format(name))
                                 sns.swarmplot(x=name, y=target, data=df);
                                 fileName = name+'SwarmPlots'+'.png'
                                 path = os.path.join(os.getcwd(),'results/SwarmPlots',fileName)
                                 plt.savefig(path)
               except:
                                 pass

        def normalQQplots(self, ypred, ytest, clf):
                        #test = np.random.normal(0,1, 1000)
                        resids = ypred - ytest
                        resids = resids/np.std(resids)
                        plt.figure()
                        plt.title('{}_Normal QQ residual plots'.format(clf))
                        sm.qqplot(resids, line='45')
                        filename = clf+'_normalQQplot.png'
                        path = os.path.join(os.getcwd(),'results',filename)
                        plt.savefig(path)

        def residVsPredPlots(self, ypred, ytest, clf):
                        #test = np.random.normal(0,1, 1000)
                        resids = ypred - ytest
                        resids = resids/np.std(resids)
                        plt.figure()
                        plt.title('residuals vs fitted values')
                        plt.scatter(ypred, resids)
                        plt.xlabel('fitted values')
                        plt.ylabel('standardised residuals')
                        filename = clf+'residVsPredPlots.png'
                        path = os.path.join(os.getcwd(),'results',filename)
                        plt.savefig(path)


        def ROCplots(self, ytest, ypred_prob, clf):
                    #print('ypred_prob is -------->{}'.format(ypred_prob))
                    fpr, tpr, thresholds = roc_curve(ytest.values, ypred_prob)
                    # create plot
                    plt.figure()
                    plt.plot(fpr, tpr, label='ROC curve')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.xlim([-0.02, 1])
                    plt.ylim([0, 1.02])
                    plt.legend(loc="lower right")
                    fileName = clf+'_ROCplot.png'
                    path = os.path.join(os.getcwd(),'results',fileName)
                    plt.savefig(path)
                    print('---------------------------------------------')
                    print('roc_auc_score :----------->{}'.format(roc_auc_score(ytest, ypred_prob)))


        def precVsRecall(self, ytest, ypred_prob, clf):
                    #print('ypred_prob is -------->{}'.format(ypred_prob))
                    precision, recall, thresholds = precision_recall_curve(ytest.values, ypred_prob)
                    # create plot
                    plt.figure()
                    plt.plot(precision, recall, label='Precision-recall curve')
                    plt.xlabel('Precision')
                    plt.ylabel('Recall')
                    plt.title('Precision-recall curve')
                    plt.legend(loc="lower left")
                    fileName = clf+'_precVsRecall.png'
                    path = os.path.join(os.getcwd(),'results',fileName)
                    plt.savefig(path)
                    print('average_precision_score :----------->{}'.format(average_precision_score(ytest, ypred_prob)))




        def plot_confusion_matrix(self, clf, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
                    """
                    This function prints and plots the confusion matrix.
                    Normalization can be applied by setting `normalize=True`.
                    """
                    if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        print("Normalized confusion matrix")
                    else:
                        print('Confusion matrix, without normalization')

                    print(cm)

                    plt.imshow(cm, interpolation='nearest', cmap=cmap)
                    plt.title(title)
                    plt.colorbar()
                    tick_marks = np.arange(len(classes))
                    plt.xticks(tick_marks, classes, rotation=45)
                    plt.yticks(tick_marks, classes)

                    fmt = '.2f' if normalize else 'd'
                    thresh = cm.max() / 2.
                    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                        plt.text(j, i, format(cm[i, j], fmt),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.tight_layout()
                    filepath = '{}.png'.format(clf)
                    path = os.path.join(os.getcwd(),'results',filepath)
                    plt.savefig(path)
