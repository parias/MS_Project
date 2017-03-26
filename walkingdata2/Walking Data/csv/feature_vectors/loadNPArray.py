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

fv_sizes = []
knn = KNeighborsClassifier(n_neighbors=5)


"""
X: Feature matrix (all the FVs)
y: response vector (what it should be)
"""
def fitKNN():
    # TODO
    print(knn)
    print(fv_sizes)

    # df = pd.read_csv('./normalized/all_user.csv') # Using Panda
    df = np.loadtxt('./normalized/all_user.csv', delimiter=',')
    # print(df)

    # Create response vector
    response_vector = []
    for i, val in enumerate(fv_sizes):
        for x in range(val):
            response_vector.append(i)
    # print(len(response_vector))
    # print(len(df))
    # knn.fit(df, response_vector)

    # http://scikit-learn.org/stable/modules/cross_validation.html
    # KNN with Cross Validation, works?
    X_train, X_test, y_train, y_test = train_test_split(df, response_vector, test_size=.3, random_state=0)
    print('Shapes', X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape)
    clf = knn.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def load_fv():
    allFV = np.empty([9])
    files = load_files()
    # return(load_np_array(allFV, files))
    allFVs, fvSizes = load_np_array(allFV, files)
    # print(type(allFVs), h)

    global fv_sizes
    fv_sizes = fvSizes
 
    normalize_fv(allFVs, fvSizes)


def normalize_fv(allFVs, fvSizes):
    # TODO
    # print(allFVs.shape)
    max_abs_scaler = p.MaxAbsScaler()
    min_scaler = p.MinMaxScaler((-1, 1))
    normalized = min_scaler.fit_transform(allFVs)

    all_normalized = []
    index = 0

    normalized_df = pd.DataFrame(normalized)
    split_normalization(normalized_df, fvSizes)


def split_normalization(normalized, fvSizes):

    users_normalized = []

    # Each Array position is np.array of all FV's of specific user
    count = 0
    for i in fvSizes:
        # user index = normalized.splice(number of FV)
        users_normalized.append(np.array(normalized[count:count+i]))
        count += i
    # print(len(users_normalized))

    # what does this do?
    # all_users_df = pd.DataFrame(users_normalized)
    
    df = pd.DataFrame()
    header = ['subject', 'mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'mad_x', 'mad_y', 'mad_z']

    for i, val in enumerate(users_normalized):
        # Adds Subject Column
        data = pd.DataFrame(val)
        data.insert(0, 'subject', i+1)

        # pd.DataFrame(val).to_csv('./normalized/' + str(i+1) + '_with_subject.csv', header=False, index=False)
        data.to_csv('./normalized/' + str(i+1) + '_with_subject.csv', header=header, index=False)
        df = df.append(data, ignore_index=True) # works correctly

    # Numpy does not like headers
    df.to_csv('./normalized/all_user_with_subject.csv', header=header, index=False)


def load_np_array(allFV, files):
    # print(files)
    fvSizes = []
    count = 0
    for f in files:
        fvFile = np.loadtxt(f, delimiter=',')
        temp_count = count_fv(fvFile)
        fvSizes.append(temp_count)
        count += temp_count
        allFV = np.vstack((allFV, fvFile))
    # print(count)
    # return allFV
    allFV = np.delete(allFV, 0, 0)
    header = ['Mean X', 'Mean Y', 'Mean Z', 'STD X', 'STD Y', 'STD Z', 'MAD X', 'MAD Y', 'MAD Z']
    # pd.DataFrame(allFV).to_csv("all_users.csv", header=header, index=False)
    return allFV, fvSizes


def load_files():
    files = []
    for f in glob.glob('*.csv'):
        files.append(f)
    return sort_nicely(files)


def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    return [try_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    return sorted(l, key=alphanum_key)


def count_fv(fvFile):
    # print(fvFile.shape)
    return fvFile.shape[0]


if __name__ == "__main__":
    load_fv()
    fitKNN()
    # print(allFV)
    # split_normalization()
