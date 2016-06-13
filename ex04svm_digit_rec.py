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
    pca2 = PCA(n_components=v+1)
    X = pca2.fit_transform(X)
    X_test = pca2.transform(X_test)
    print('*** PCA: reached variance: %2.4f, Numb. Dim.: %2.0f ***' % (reached_var, v+1))
    return X,X_test, reached_var, v+1


def classifiy(X_train, y_train, X_test, y_test, n):
    print('Building classifier...')
    clf = SVC()
    clf.set_params(kernel='poly', C=1e-14, degree=3, coef0=-1e-90)
    clf.fit(X_train, y_train)
    print('Predicting...')
    
    img_num = randint(0, len(y_test)-1)
    prediction = clf.predict(X_test[img_num:(img_num+n)])
    print('Score of classifier on training data: \n %1.3f' % 
          (clf.score(X_train, y_train)))
    print('Score of classifier on test data: \n %1.3f' % 
          (clf.score(X_test, y_test)))
    print('Predicted labels:\n', prediction)
    print('Real labels:\n',y_test[img_num:(img_num+n)])
    

def k_cross_val(Data, label, k):    # (Data Matrix, Label Vector, k)
    mode = 1    # grid search off: 0; Grid-Search on: 1

    kf = KFold(len(Data), n_folds=k)
    # Grid-Search parameters
    param_grid = [{'C': [10], 
                    'gamma': [1e-07], 'kernel': ['rbf']} 
                    ,{'C': [1e-14], 'degree': [3], 'coef0':[-1e-90],
                    'kernel': ['poly']}]
    print('Grid-Search with Cross Validation...')
    score_results = []
    for train_index, test_index in kf:
        # create datasets from cross validation
        X_train = np.array([Data[tr_i] for tr_i in train_index])
        y_train = np.array([label[tr_i] for tr_i in train_index])
        X_test = np.array([Data[te_i] for te_i in test_index])
        y_test = np.array([label[te_i] for te_i in test_index])

        # Create SVM classifier and fit to data
        if mode == 0:
            clf = SVC()
            clf.set_params(kernel='poly', C=1e-14, coef0=-1e-90)
        elif mode == 1:
            svr = SVC()
            clf = grid_search.GridSearchCV(svr, param_grid, n_jobs=4)
        clf.fit(X_train, y_train)
        print('Best parameters found: ', clf.best_params_)
        print('Score with these parameters: %2.5f' % (clf.best_score_))
        # predict labels
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        # evaluate SVM
        s = clf.score(X_test, y_test)
        score_results.append(s)
    mean_score=sum(score_results)/len(score_results)
    print('Mean Score on Test Data: %2.5f' % (mean_score))
    return mean_score


def main():
    mode = 1    # (0):Cross-Val with Grid Search
                # (1):Testing the classifier
    mode_text = ['Cross-Val with Grid Search', 'Testing the classifier']
    print('***** Mode %1.0i - %s ******' % (mode, mode_text[mode]))
   
    k = 5    # k-fold Cross-Validation (mode (0))
    var_increasing = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, .99]    # (mode (0))
    var_desired = 0.9    # desired variance (only mode (1))
    n = 100    # Test classifier on n random test images (mode (1))

    # load data
    print('Loading Data...')
    train_data, train_labels ,test_data, test_labels = load_data() 

    # Mode (0):  Cross-Val with Grid Search
    if mode == 0:    
        mean = []
        # cross validation with increasing variance
        for var in var_increasing:
            print('*****************************************************')
            print('Dimensionality reduction  (desired variance %1.2f)' % (var))
            X_train_pca, X_test_pca ,reached_var, v = pca_dim_reduction(train_data,
                                                                         test_data, var)
            print('*****************************************************')
            mean_score = k_cross_val(X_train_pca, train_labels, k)
            mean.append(mean_score)
        # plot result
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(var_increasing, mean, c='blue')
        ax1.set_title('Score over Variance of SVM')  
        ax1.set_xlabel('Variance [%]')
        ax1.set_ylabel('Mean Score of k-fold Cross-Validation') 
    # Mode (1): Testing the classifier
    elif mode == 1:    
        print('Dimensionality reduction  (desired variance %1.2f)' % (var_desired))
        X_train_pca, X_test_pca ,reached_var, v = pca_dim_reduction(train_data, 
                                                                    test_data, var_desired)
        # Train SVM with previously determined hyperparameters
        classifiy(X_train_pca, train_labels, X_test_pca, test_labels, n)

      
if __name__ == "__main__":    # If run as a script, create a test object
    main()
    #plt.show() 
