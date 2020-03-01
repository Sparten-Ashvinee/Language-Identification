#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:13:42 2018

@author: ashvinee
"""

file = 'test-eng.txt'
file1 = 'eng-train.txt'
file2 = 'eng-test.txt'

fin = open(file, 'r', encoding='utf8')
fin1 = open(file1, 'w', encoding='utf8')
fin2 = open(file2, 'w', encoding='utf8')

c=0
for line in fin:
    str = 'EN'
    line = line[:-1]
    line += '\t'
    line += str
    line += '\n'
    print(line)
    c=c+1
    if c <= 800:
        fin1.write(line)
    elif c > 800:
        fin2.write(line)

print(c)