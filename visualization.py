import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
class Visualisation():
        def __init__(self, df = None ,target = None):
            #file_load.__init__(self)
            self.target = target
            self.df = df
        
        def plot(self):
            df = self.df
            target = self.target
            
            plt.figure()
            plt.hist(df[target])
            plt.show()
            
        def correlationPlot(self):
            df = self.df

            self.names = [i for i in df.columns if df[i].dtype.name == 'float64']
            corr = df[self.names].corr()
            
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            
        def wordCloud(self, allReviews):
            i = 1
            for cat in allReviews.keys():
                wordcloud = WordCloud().generate(allReviews[cat])
                plt.figure(i)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title('{}'.format(cat))
                plt.show()
                i+= 1
                    
                
        def silhouetteScores(self, scores, clusterSizes):
            plt.figure()
            plt.plot(clusterSizes,scores)
            plt.xlabel('cluster size')
            plt.ylabel('average silhouette scores')
            plt.title('silhouette scores vs cluster size')
            plt.show()
            
            