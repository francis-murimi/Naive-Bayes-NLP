from joblib import load
from sklearn.model_selection import train_test_split

from prepare_input import ndf
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
