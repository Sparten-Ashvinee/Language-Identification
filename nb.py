#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:56:50 2018

@author: ashvinee
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Each line is spilted into sentences and labels
def data(filename):
    with open(filename, 'r', encoding='utf8') as fin:
        sentences, labels = [], []
        for line in fin:
            x, y = line.strip().split('\t')
            sentences.append(x)
            if y:
                labels.append(y)
        return sentences, labels

# eng-train.txt dataset is used for training the models
#this file is passed to data function
X_train, y_train = data('eng-train.txt')
# eng-test.txt dataset is used for testing
#this file is passed to data function
X_dev, y_dev = data('eng-test.txt')

#Using CountVectorizer for future extraction from text in training dataset
#print ('Vectorizing...', flush=True)
ngram_vectorizer = CountVectorizer(analyzer='char',ngram_range=(2, 2), min_df=1)
trainset = ngram_vectorizer.fit_transform(X_train)
tags = y_train

#Training model using MultinomialNB ml algo
#print ('Working on MultinomialNB... ', flush=True)
NBclassifier = MultinomialNB()
#trainset contains vectorize sentences
#tags contains labels
NBclassifier.fit(trainset, tags)

#Using CountVectorizer for future extraction from text in testing dataset
devset = ngram_vectorizer.transform(X_dev)
predictions = NBclassifier.predict(devset)
#Calculating prediction accuracy score from testing dataset
print (accuracy_score(y_dev, predictions))
'''
#Displaying predicted language identified
#for p in predictions:
#    print(p)

print("cross_val_predict...")
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(NBclassifier,trainset, tags,cv=3)

print("Calculating precision and recall score...")
from sklearn.metrics import precision_score, recall_score
pscore = precision_score(tags,y_train_pred,average='micro')
print("precision_score : ", pscore)
rscore = recall_score(tags,y_train_pred,average='micro')
print("recall_score : ", rscore)

print("Calculating f1_score...")
from sklearn.metrics import f1_score
fscore = f1_score(tags,y_train_pred,average='micro')
print("f1_score : ", fscore)
'''
#True-Positive
TP = 0
#False-Positive
FP = 0
#True-Negative
TN = 0
#False-Negative
FN = 0

for i in range(len(predictions)): 
    #y_dev[i] contains actual labels
    #predictions[i] contains predicted labels
    if y_dev[i]==predictions[i]=='EN':
       TP += 1
    if y_dev[i]!='EN' and predictions[i]=='EN':
       FP += 1
    if y_dev[i]=='EN' and predictions[i]!='EN':
       TN += 1
    if y_dev[i]!='EN' and predictions[i]!='EN':
       FN += 1
       
print(TP)
print(FP)
print(TN)
print(FN)
'''
#import numpy as np
y_true = ([])
for i in predictions:
    if i=='EN':
        y_true.append(1)
    else:
        y_true.append(0)
y_true.toarray()

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(y_true, y_score)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

'''
#predicting language presented in eng-predict.txt files
prefile = 'eng-predict.txt'
pf = open(prefile, 'r', encoding='utf8')
for line in pf:
    #Converting each line into list
    dline = [line]
    #Vactorizing
    strmod = ngram_vectorizer.transform(dline)
    #predicting lanhuahe of sentence
    predvalue = NBclassifier.predict(strmod)
    #print(predvalue)

#Displaying whether given sentence is valid or not
if predvalue == 'EN':
    print("Valid")
else:
    print("Invalid")