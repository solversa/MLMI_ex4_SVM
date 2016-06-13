#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import matplotlib
import matplotlib.patches as mpatches



def svm_fit(data_mat, class_vect, kernel, c_val, gamma):
    # Possible Kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    clf = SVC()
    clf.set_params(kernel=kernel, C=c_val, gamma=gamma)
    clf.fit(data_mat, class_vect)
    return clf


def plot_predict_vs_true_values(data, predicted_classes, true_classes):
    classified_data = [[],[]]    # [0]: correct classified, [1]: missclassified
    for i in range(predicted_classes):
        if predicted_classes[i] == true_classes[i]:
            classified_data[0].append(data[i])
        else:
            classified_data[1].append(data[i])
    plt.scatter(classified_data[0])



def load_data(file_name_x_train, file_name_y_train,
              file_name_x_test, file_name_y_test):
    # get path
    path = os.getcwd()
    # read data
    series_x_test = pd.read_csv(path+'/twomoons/'+file_name_x_test, header=None)
    series_y_test = pd.read_csv(path+'/twomoons/'+file_name_y_test, header=None)
    series_x_train = pd.read_csv(path+'/twomoons/'+file_name_x_train, header=None)
    series_y_train = pd.read_csv(path+'/twomoons/'+file_name_y_train, header=None)
    #data_x_test = np.array([series_x_test.values[:,0], series_x_test.values[:,1]])
    #data_x_train = np.array([series_x_train.values[:,0], series_x_train.values[:,1]])
    X_train = series_x_train.values
    y_train = series_y_train.values.ravel()
    X_test = series_x_test.values
    y_test = series_y_test.values.ravel()
    return X_train, y_train, X_test, y_test


### Exercise 1 SVMs on Toy Dataset ###
def test_classifier(X_train, y_train, X_test, y_test):
    # Set testing parameters
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    c_val = [0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 200, 500, 1000, 10000]
    gamma = [0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 200, 500, 1000, 10000]
    
    # Test with Linear Kernel
    mse_train = []
    mse_test = []
    score_train = []
    score_test = []
    for c in c_val:
        # creat classifier
        clf = svm_fit(X_train, y_train, kernel='linear', c_val=c, gamma=0.1)    
        # predict classes on training data and test data
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        # score of classifier on training and test data
        score_train.append(clf.score(X_train, y_train))
        score_test.append(clf.score(X_test, y_test))
        # mse between predictet and real classes of training and test data
        mse_train.append(((y_train - y_train_pred) ** 2).mean())
        mse_test.append(((y_test - y_test_pred) ** 2).mean())

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(211)
    ax2.plot(c_val, mse_train, c='blue')
    ax2.plot(c_val, mse_test, c='red')          
    ax2.set_xscale('log') 
    ax2.set_title('MSE on SVM with linear kernel')  
    ax2.set_xlabel('C-Value')
    ax2.set_ylabel('MSE')    
    blue_patch = mpatches.Patch(color='blue', label='Training Data')
    red_patch = mpatches.Patch(color='red', label='Test Data')
    ax2.legend(handles=[red_patch, blue_patch], loc=1)
    ax21 = fig2.add_subplot(212)
    ax21.plot(c_val, score_train, c='blue')
    ax21.plot(c_val, score_test, c='red')          
    ax21.set_xscale('log') 
    ax21.set_title('Score on SVM with linear kernel')  
    ax21.set_xlabel('C-Value')
    ax21.set_ylabel('Score')
    ax21.legend(handles=[red_patch, blue_patch], loc=4)


    # Test with RBF Kernel
    mse_train = []
    mse_test = []
    score_train = []
    score_test = []
    for c in c_val:
        # creat classifier
        clf = svm_fit(X_train, y_train, kernel='rbf', c_val=c, gamma=0.1)    
        # predict classes on training data and test data
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        # score of classifier on training and test data
        score_train.append(clf.score(X_train, y_train))
        score_test.append(clf.score(X_test, y_test))
        # mse between predictet and real classes of training and test data
        mse_train.append(((y_train - y_train_pred) ** 2).mean())
        mse_test.append(((y_test - y_test_pred) ** 2).mean())

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(211)
    ax3.plot(c_val, mse_train, c='blue')
    ax3.plot(c_val, mse_test, c='red')          
    ax3.set_xscale('log') 
    ax3.set_title('MSE on SVM with RBF kernel')  
    ax3.set_xlabel('C-Value')
    ax3.set_ylabel('MSE')    
    blue_patch = mpatches.Patch(color='blue', label='Training Data')
    red_patch = mpatches.Patch(color='red', label='Test Data')
    ax3.legend(handles=[red_patch, blue_patch], loc=1)
    ax31 = fig3.add_subplot(212)
    ax31.plot(c_val, score_train, c='blue')
    ax31.plot(c_val, score_test, c='red')          
    ax31.set_xscale('log') 
    ax31.set_title('Score on SVM with RBF kernel')  
    ax31.set_xlabel('C-Value')
    ax31.set_ylabel('Score')
    ax31.legend(handles=[red_patch, blue_patch], loc=4)
    comparing_plot(X_test, y_test, y_test_pred)



def comparing_plot(X, y, y_pred):
    y_delta = y + (y-y_pred) * 10
    colors_plot = ['orange', 'blue', 'green', 'red']
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    ax4.scatter(X[:,0], X[:,1], c=y_delta,
                cmap=matplotlib.colors.ListedColormap(colors_plot))


def main():
    # data file names
    X_train, y_train, X_test, y_test = load_data(
        'xtrain.csv', 'ytrain.csv', 'xtest.csv', 'ytest.csv')

    # Plot training Data
    colors_plot = ['red','blue']
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.scatter(X_train[:,0], X_train[:,1], c=y_train,
                cmap=matplotlib.colors.ListedColormap(colors_plot))
    ax1.set_title('Original Training Data')
    # Exercise 1    
    test_classifier(X_train, y_train, X_test, y_test)

    # create 2D plots of the points comparing the predicted vs. true values 
    clf = SVC()
    clf.fit(X_train, y_train)  
    y_test_pred = clf.predict(X_test)
    comparing_plot(X_test, y_test, y_test_pred)


if __name__ == "__main__":    # If run as a script, create a test object
    main()
    plt.show() 


    





