"""
Loads Feature Vector into Numpy Arrays


Loads from CSV files to feature vectors then Normalizes(-1 to 1)
using sklearn preprocessing module. Prints using Pandas

"""

__authors__ = ["Pablo A. Arias <paarias24@gmail.com>"]

from sklearn import preprocessing as p
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import glob
import re
import numpy as np
import pandas as pd


# training = pd.read_csv('training_data_normalized.csv')
# test = pd.read_csv('test_data_normalized.csv')

training = []
test = []
all_data = []
knn = KNeighborsClassifier(n_neighbors=5)


def load_files():
    global training, test, all_data
    with open('training_data_normalized.csv') as f:
        training = pd.read_table(f, sep=',', index_col=0)
    with open('test_data_normalized.csv') as f:
        test = pd.read_table(f, sep=',', index_col=0)
    all_data = merge_data(training, test)
    pd.DataFrame(all_data).to_csv('all_data_normalized.csv', index=False)

def merge_data(training, test):
    joined = [training, test]
    result = pd.concat(joined)
    return result.sort_index()


def load_data():
    return pd.read_csv('all_data_normalized.csv')


def classify(data):

    # print(data_frame)
       
    # http://scikit-learn.org/stable/modules/cross_validation.html
    # KNN with Cross Validation, works?
    X_train, X_test, y_train, y_test = train_test_split(data.drop('subject', axis=1).as_matrix(), data['subject'].as_matrix(), test_size=.3, random_state=0)
    print('Shapes- X_train, X_test, y_train, y_test:', X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape)
    clf = knn.fit(X_train, y_train)
    print(clf.score(X_test, y_test))



if __name__ == "__main__":
    # load_files()
    # print(all_data)
    classify(load_data())
