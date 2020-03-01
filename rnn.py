#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 19:01:28 2018

@author: ashvinee
"""

import tensorflow as tf

def dat(filename):
    with open(filename, 'r', encoding='utf8') as fin:
        sentences = []
        for line in fin:
            x, y = line.strip().split('\t')
            sentences.append(x)
        return sentences
    
def lab(filename):
    with open(filename, 'r', encoding='utf8') as fin:
        labels = []
        for line in fin:
            x, y = line.strip().split('\t')
            if y:
                labels.append(y)
        return labels
'''
X_train, y_train = data('eng-train.txt')
X_dev, y_dev = data('eng-test.txt')
'''
x_data = tf.Variable(5)
b = x_data * x_data
X_train = tf.Variable(dat('eng-train.txt'))

sess = tf.Session()
print(sess.run(b))