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

#print(ndf)