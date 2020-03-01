#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:41:33 2018

@author: ashvinee
"""

# -*- coding: utf-8 -*-

#import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

def data(filename):
    with open(filename, 'r', encoding='utf8') as fin:
        sentences, labels = [], []
        for line in fin:
            x, y = line.strip().split('\t')
            sentences.append(x)
            if y:
                labels.append(y)
        return sentences, labels

X_train, y_train = data('eng-train.txt')
X_dev, y_dev = data('eng-test.txt')

#print ('Vectorizing...', flush=True)
ngram_vectorizer = CountVectorizer(analyzer='char',ngram_range=(2, 2), min_df=1)
trainset = ngram_vectorizer.fit_transform(X_train)
tags = y_train

#SVM
#print ('Working on SVM training... ', flush=True)
SVMclssifier = svm.SVC()
SVMclssifier.fit(trainset, tags)

#print ('Working on SVM prediction... ', flush=True)
devset = ngram_vectorizer.transform(X_dev)
predictions = SVMclssifier.predict(devset)
print (accuracy_score(y_dev, predictions))

'''
#Displaying predicted language identified
for p in predictions:
    print(p)

print("cross_val_predict...")
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(SVMclssifier,trainset, tags,cv=3)

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
#predicting
prefile = 'eng-predict.txt'
pf = open(prefile, 'r', encoding='utf8')
for line in pf:
    dline = [line]
    strmod = ngram_vectorizer.transform(dline)
    predvalue = SVMclssifier.predict(strmod)
    print(predvalue)

if predvalue == 'EN':
    print("Valid")
else:
    print("Invalid")
