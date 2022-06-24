import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd

df = pd.read_csv('clean.csv')
from sklearn.model_selection import train_test_split
X = df.text
y = df.title
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.feature_extraction.text import CountVectorizer
#max_features=2000,
#vect = CountVectorizer(max_features=2000, binary=True)
vect = CountVectorizer(binary=True,max_features=2000)

X_train_vect = vect.fit_transform(X_train)

counts = df.title.value_counts()
#print(counts)

#print("\nPredicting only 0 = {:.2f}% accuracy".format(counts[0] / sum(counts) * 100))
#print("\nPredicting only 1 = {:.2f}% accuracy".format(counts[1] / sum(counts) * 100))

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_vect, y_train)

nb.score(X_train_vect, y_train)

X_test_vect = vect.transform(X_test)

y_pred = nb.predict(X_test_vect)

#print(y_pred)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
#print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
#print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))