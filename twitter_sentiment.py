#!/usr/bin/env python

"""twitter_sentiment.py: The python file classifies a million tweets into positive and negative tweets."""

__author__      = "Tariq Khaleeq"


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import operator


def prepro():
    data=[]
    target=[]

    for rows in range(0,len(twitter_data.index)):
        if twitter_data.iloc[rows,0] == 0:
            #print twitter_data.iloc[rows,0]
            target.append(0)
            data.append(twitter_data.iloc[rows,5])
        else:
            #print twitter_data.iloc[rows,0]
            target.append(1)
            data.append(twitter_data.iloc[rows,5])

    print "Preprossesing of data and target done."
    print "Ready for ML"

    return data, target

def datasplit(data):
    tf=TfidfVectorizer(min_df=0, max_features=None, encoding='utf8' , decode_error='ignore',strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
                            use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")
    tf_twitter_data=tf.fit_transform(data)

    print twitter_data.shape
    print ('Text Feature Extracted looks like:',tf_twitter_data.shape)

    return tf_twitter_data

def trainmodel(tf_twitter_data, target):
    model=LogisticRegression(C=1)
    model.fit(tf_twitter_data,target)
    positive_model=model.predict_proba(tf_twitter_data)[:,1]
    return positive_model

def auctest(target,positive_model):
    print (" auc " , roc_auc_score(target,positive_model))

def display(data,positive_model):
    array_with_all_elements=[]
    for i in range (len(positive_model)):
        array_with_all_elements.append([data[i],positive_model[i] ])

    array_with_all_elements=sorted(array_with_all_elements, key=operator.itemgetter(1))

    print ("Top negative comments")
    for i in range (10):
        print ("probability: %f negative comment text: %s" % (array_with_all_elements[i][1],array_with_all_elements[i][0]))

    print ("top positive comments")
    for i in range (len(array_with_all_elements)-1,len(array_with_all_elements)-11,-1 ):
        print ("probability: %f positive comment text: %s" % (array_with_all_elements[i][1],array_with_all_elements[i][0]))


if __name__== '__main__':
    print 'reading twitter data input'
    #readinput()
    twitter_data=pd.read_csv('/Users/tariqkhaleeq/Downloads/training.1600000.processed.noemoticon.csv')
    print ("data file dim: ",twitter_data.shape)
    data,target=prepro()
    tf_twitter_data=datasplit(data)
    positive_model=trainmodel(tf_twitter_data, target)
    auctest(target,positive_model)
    display(data,positive_model)
