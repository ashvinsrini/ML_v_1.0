import nltk
import pickle
#from classificationModel import SVM, Logistic, RandomForest, NeuralNetwork
import random
import os
#import pandas as pd
#from sklearn.metrics import accuracy_score
class SentimentAnalysis():
    def __init__(self):
        #SVM.__init__(self, target = None, df = None)
        pass

class SentimentTrain(SentimentAnalysis):
    def __init__(self):
        pass

    def extract(self, reviews, catNum, catTypes, label):
        allReviews = {}
        i = 0
        for cat in catTypes:
            allReviews[cat] = open(reviews[i],"r", encoding='latin2').read()
            i+= 1
        ####### Documents have each word entries with its associated category ######

        def split(rev,cat,docs):
            for ii in rev.split('\n'):
                docs.append((ii, cat))
            return docs
        documents = []

        for cat in catTypes:
            documents = split(allReviews[cat], cat, documents)

        def partsOfSpeechExtract(wordList):
            allWordTypes=['J','V','N']
            retList=[]
            temp=nltk.pos_tag(wordList)
            for w in temp:
                if (w[1][0]) in allWordTypes:
                    retList.append(w[0].lower())
            return retList

        ####### pos_reviews_words have words  from positive text and likewise for neg_reviews_words

        reviewWords = {}
        importantWords = {}
        for cat in catTypes:
            reviewWords[cat] = nltk.word_tokenize(allReviews[cat])
            importantWords[cat] = partsOfSpeechExtract(reviewWords[cat])
        ##### all_words contain the set of words occuring in all documents ########

        allWords=[]
        for cat in catTypes:
            for word in importantWords[cat]:
                allWords.append(word.lower())
        ####### all_words_freq contains all words with its associated frequencies #######
        allWordsFreq=nltk.FreqDist(allWords)
        wordFeats=list(allWordsFreq.keys())[:5000]

        ''' find_feats function returns true if each word of the 5000 words is present
        in the document else false. So every features returned is a 5000 vector with
        true or false entries
        '''

        #featuresList = 'word_feats.pickle'
        #filepath = os.path.join(os.getcwd(),'results',featuresList)
        #temp_var=open(filepath,'wb')
        #pickle.dump(wordFeats,temp_var)
        #temp_var.close()


        def findFeats(document, wordFeats):
            words=nltk.word_tokenize(document)
            words=partsOfSpeechExtract(words)
            features={}
            for w in wordFeats:
                 features[w]=w in words
            return features

        ### Creating feature list for each document
        features_list=[(findFeats(review, wordFeats),category) for (review,category) in documents]
        random.shuffle(features_list)
        trainSet=features_list[:10000]
        testSet=features_list[10000:]

        fileTrainName = 'train_data.pickle'
        filepath = os.path.join(os.getcwd(),'results',fileTrainName)
        temp_var=open(filepath,'wb')
        pickle.dump(trainSet,temp_var)
        temp_var.close()

        fileTestName = 'test_data.pickle'
        filepath = os.path.join(os.getcwd(),'results',fileTestName)
        temp_var=open(filepath,'wb')

        pickle.dump(testSet,temp_var)
        temp_var.close()
        classifier = nltk.NaiveBayesClassifier.train(trainSet)
        modelName = 'model.pickle'
        filepath = os.path.join(os.getcwd(),'results',modelName)
        temp_var=open(filepath,'wb')
        pickle.dump(classifier,temp_var)
        temp_var.close()


        accuracy = nltk.classify.accuracy(classifier, testSet)*100
        print("Original Naive Bayes Algo accuracy percent:",(accuracy))
        return accuracy, allReviews

class SentimentTest(SentimentAnalysis):
    def __init__(self):
        pass
