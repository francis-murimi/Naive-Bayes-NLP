import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('clean.csv')
X = df.text
y = df.title
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#max_features=2000,
#vect = CountVectorizer(max_features=2000, binary=True)
vect = CountVectorizer(binary=True,max_features=2000, vocabulary = None)

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


def classify(filename):
    #filename = 'safe-chicken.txt'
    with open(filename) as f:
        contents = f.read()
        f.close()

    from nltk.tokenize import word_tokenize

    data = word_tokenize(contents.lower())

    from nltk.corpus import stopwords

    english_stopwords = stopwords.words('english')
    t_data = [t for t in data if t not in english_stopwords]

    from nltk import PorterStemmer

    ps = PorterStemmer()
    token_data = [ps.stem(i) for i in t_data]

    # Join tokenized words
    token_text = ' '.join([str(item) for item in token_data])

    data = {'text': token_text,
            'title': None}
    # Create DataFrame
    data_df = pd.DataFrame(data, index= [0])

    ndf = pd.DataFrame(np.repeat(data_df.values, 3, axis=0))
    ndf.columns = data_df.columns

    A = ndf.text
    b = ndf.title
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.30)

    A_test_vect = vect.transform(A_test)


    my_pred = nb.predict(A_test_vect)

    if my_pred == 1:
        print('Agriculture')
    else:
        print('Not agriculture')


classify('safe-chicken.txt')

