#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import matplotlib
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA


def load_data():
    # get path
    path = os.getcwd()
    # read data
    series_train = pd.read_csv(path+'/04-digits-dataset/train_small.csv',
                               header=None, low_memory=False)
#    series_test = pd.read_csv(path+'/04-digits-dataset/test.csv', 
#                              header=None, low_memory=False)
#    series_out = pd.read_csv(path+'/04-digits-dataset/out.csv', 
#                             header=None, low_memory=False)

    train_labels = np.array(series_train[0][1:])
    train_data = series_train[1:]
    train_data = train_data.drop(0, 1)


    return train_data.values.astype(int), train_labels
          #test , out


def pca_dim_reduction(X, desired_var):
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
    return X, reached_var, v


def main():
    # load data
    train_data, train_labels = load_data() 
    
    # dimensionality reduction using pca with svd
    X_after_pca, reached_var, v = pca_dim_reduction(train_data, 0.8)
    print('x transformed shape: ',X_after_pca.shape)
    print('reached variance:',reached_var)
    print('number of needed principle components', v)


    





if __name__ == "__main__":    # If run as a script, create a test object
    main()
    plt.show() 
