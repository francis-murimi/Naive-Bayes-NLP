import numpy as np
import pandas as pd

filename = 'sakaja.txt'    
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
df = pd.DataFrame(data, index= [0])

ndf = pd.DataFrame(np.repeat(df.values, 3, axis=0))
ndf.columns = df.columns

from joblib import load
from sklearn.model_selection import train_test_split

from train_model import vect

my_nb = load('nbmodel.joblib')

A = ndf.text
b = ndf.title
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.30)

A_test_vect = vect.transform(A_test)


my_pred = my_nb.predict(A_test_vect)

if my_pred == 1:
    print('Agriculture')
else:
    print('Not agriculture')

print(my_pred)
