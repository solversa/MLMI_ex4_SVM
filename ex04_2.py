#!/usr/bin/env python3
'''
Digit Recognition using Support Vector Machines with k-fold Cross Validation
on the MNIST Dataset (containing 42000 images of size 28x28 of hand written digits)
'''

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from random import randint


def load_data():
    # get path
    path = os.getcwd()
    # read data
    series_train = pd.read_csv(path+'/04-digits-dataset/train.csv',
                               header=None, low_memory=False)
    series_test = pd.read_csv(path+'/04-digits-dataset/test.csv', 
                              header=None, low_memory=False)
    series_out = pd.read_csv(path+'/04-digits-dataset/out.csv', 
                             header=None, low_memory=False)

    train_labels = np.array(series_train[0][1:])
    train_data = series_train[1:]
    train_data = train_data.drop(0, 1)
    test_data = series_test.values[1:].astype(int)
    test_labels = series_out[1].values[1:].astype(int)
    return train_data.values.astype(int), train_labels.astype(int), test_data, test_labels


def pca_dim_reduction(X, X_test, desired_var):
    # Check how many principle components needed for desired variance
    pca = PCA()
    pca.fit(X)
    reached_var = 0 
    for v in range(len(pca.explained_variance_ratio_)):
        reached_var = reached_var + pca.explained_variance_ratio_[v]
        if reached_var >= desired_var:
            break
    # PCA with desired number ov principle components
    pca2 = PCA(n_components=v)
    X = pca2.fit_transform(X)
    X_test = pca2.fit_transform(X_test)
    return X, X_test, reached_var, v


def classifiy(X_train, y_train, X_test, y_test, n):
    print('Building classifier...')
    clf = SVC()
    clf.set_params(kernel='rbf', C=10, gamma=1e-07)
    clf.fit(X_train, y_train)
    print('Predicting...')
    
    img_num = randint(0, len(y_test)-1)
    prediction = clf.predict(X_test[img_num:(img_num+n)])
    print('*******************************************')
    print('Score of classifier on test data: \n', clf.score(X_test, y_test))
    print('Predicted and real labels:\n', prediction, y_test[img_num:(img_num+n)])
    

def k_cross_val(Data, label, k):    # (Data Matrix, Label Vector, k)
    kf = KFold(len(Data), n_folds=k)
    # Grid-Search parameters
    param_grid = [ {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear']},
                   {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
                    'gamma': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 100], 'kernel': ['rbf']}, ]

    print('Gridsearch with Cross Validation...')
    print('*******************************************')
    for train_index, test_index in kf:
        # create datasets from cross validation
        X_train = np.array([Data[tr_i] for tr_i in train_index])
        y_train = np.array([label[tr_i] for tr_i in train_index])
        X_test = np.array([Data[te_i] for te_i in test_index])
        y_test = np.array([label[te_i] for te_i in test_index])

        # Create SVM classifier and fit to data
        svr = SVC()
        clf = grid_search.GridSearchCV(svr, param_grid)
        clf.fit(X_train, y_train)
        print('Best parameters found: ', clf.best_params_)
        # predict labels
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        # evaluate SVM
        print('Score on Test Data: %2.5f' % (clf.score(X_test, y_test)))


def main():
    # load data
    print('Loading Data...')
    train_data, train_labels, test_data, test_labels = load_data() 

    print('Dimensionality reduction...')
    # dimensionality reduction using pca with svd
    X_train_pca, X_test_pca, reached_var, v = pca_dim_reduction(train_data,
                                                                test_data, 0.8)
    print('*******************************************')
    print('x train pca shape: ',X_train_pca.shape)
    print('x test pca shape: ',X_test_pca.shape)
    print('reached variance:',reached_var)
    print('number of needed principle components', v)
    print('*******************************************')

    # k-fold cross validation
    k = 5
    #k_cross_val(X_train_pca, train_labels, k)

    # Train SVM with previously determined hyperparameters
    n = 10    # n random test images classified
    classifiy(X_train_pca, train_labels, X_test_pca, test_labels, n)


if __name__ == "__main__":    # If run as a script, create a test object
    main()
    plt.show() 
