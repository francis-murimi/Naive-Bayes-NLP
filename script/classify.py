import numpy as np
#import pandas as pd
#import math
#import random
#from collections import defaultdict
#from pprint import pprint

#from sklearn.naive_bayes import MultinomialNB
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from prepare_input import ndf
from train_model import nb 

my_nb = load('nbmodel.joblib')

A = ndf.text
b = ndf.title
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.30)

print('splitting')

vect = CountVectorizer(binary=True,max_features=2000)

print('vect')
A_test_vect = vect.transform(A_test)

print('transforming')

my_pred = nb.predict(A_test_vect)

if my_pred == 1:
    print('Agriculture')
else:
    print('Not agriculture')
