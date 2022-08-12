# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:00:12 2021

@author: ballo
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import KFold, GridSearchCV
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import dask as dd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, matthews_corrcoef 
import sklearn
from sklearn import svm, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


predictors = [ensemble.RandomForestClassifier( random_state = 1, n_estimators = 100),ensemble.GradientBoostingClassifier(),
              KNeighborsClassifier(n_neighbors=10, n_jobs = 5, weights = 'distance' ), MLPClassifier()]

def get_rar_dataset(filename, n=None):


    with open(filename) as data_file:
        reader = csv.reader(data_file)
        names = np.array(list(next(reader)))

    data = pd.read_csv(filename, dtype=object)
    data = data.to_numpy()

    n = len(names) - 1

    # ## Extract data names, membership values and Gram matrix

    names = names[1:n+1]
    mu = np.array([float(row[0]) for row in data[0:n+1]])
    gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                     for row in data[0:n+1]])

    assert(len(names.shape) == 1)
    assert(len(mu.shape) == 1)
    assert(len(gram.shape) == 2)

    assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

    X = np.array([[x] for x in np.arange(n)])

    return X, gram, mu, names





file_name='mirrorcompareclassificationmatrixowlthing'

X, gram, mu, names = get_rar_dataset("/corese/axiom-prediction/fragments/"+file_name+".csv")
print('done extracting matrix')

print(file_name)

for i in range(1):

     X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.7, stratify=mu)

     train_test = gram[X_train.flatten()][:, X_train.flatten()]
     test_test = gram[X_test.flatten()][:, X_train.flatten()]
     test_names = names[X_test.flatten()]
     #crossval = cross_val_score(rs, gram[X.flatten()][:, X.flatten()], mu, cv=5)
     ticfirst = time.perf_counter()
     predictors[0].fit(train_test, mu_train)
     predictors[1].fit(train_test, mu_train)
     predictors[2].fit(train_test, mu_train)
     predictors[3].fit(train_test, mu_train)
     tocfirst = time.perf_counter()
     print(f"it took {tocfirst - ticfirst:0.4f} seconds")
     predicted_test_1 = predictors[0].predict(test_test)
     predicted_test_2 = predictors[1].predict(test_test)
     predicted_test_3 = predictors[2].predict(test_test)
     predicted_test_4 = predictors[3].predict(test_test)
     print("score: 1 " ,  matthews_corrcoef (mu_test, predicted_test_1))
     print("score: 2 " ,  matthews_corrcoef (mu_test, predicted_test_2))
     print("score: 3 " ,  matthews_corrcoef (mu_test, predicted_test_3))
     print("score: 4 " ,  matthews_corrcoef (mu_test, predicted_test_4))
     # predict_train= rs.predict(train_test)
     print(f'fold {i}:')
     print('test 1')
     print(classification_report(mu_test,  predicted_test_1))
     print('test 2')
     print(classification_report(mu_test,  predicted_test_2))
     print('test 3')
     print(classification_report(mu_test,  predicted_test_3))
     print('test 4')
     print(classification_report(mu_test,  predicted_test_4))
     # print('train')
     # print(classification_report(mu_train,  predict_train))
     min_proba1 = predictors[0].predict_proba(test_test)
     min_proba2 = predictors[1].predict_proba(test_test)
     min_proba3 = predictors[2].predict_proba(test_test)
     min_proba4 = predictors[3].predict_proba(test_test)
     #print(crossval)
     #print("%0.2f accuracy with a standard deviation of %0.2f" % (crossval.mean(), crossval.std()))
     ConfusionMatrixDisplay(confusion_matrix(mu_test, predicted_test_1),display_labels=['Rejected','Accepted']).plot()
     ConfusionMatrixDisplay(confusion_matrix(mu_test, predicted_test_2),display_labels=['Rejected','Accepted']).plot()
     ConfusionMatrixDisplay(confusion_matrix(mu_test, predicted_test_3),display_labels=['Rejected','Accepted']).plot()
     ConfusionMatrixDisplay(confusion_matrix(mu_test, predicted_test_4),display_labels=['Rejected','Accepted']).plot()
     
full_view_min = np.concatenate([np.vstack((test_names,mu_test, predicted_test_1,predicted_test_2,predicted_test_3,predicted_test_4)).T,min_proba1,min_proba2,min_proba3,min_proba4], axis = 1)
wrong_predictions_min = full_view_min[(full_view_min[:,1] != full_view_min[:,2]) | (full_view_min[:,1] != full_view_min[:,3]) |(full_view_min[:,1] != full_view_min[:,4]) | (full_view_min[:,1] != full_view_min[:,5]) ]
correct_predictions_min = full_view_min[(full_view_min[:,1] == full_view_min[:,2]) | (full_view_min[:,1] != full_view_min[:,3]) |(full_view_min[:,1] != full_view_min[:,4]) | (full_view_min[:,1] != full_view_min[:,5]) ]
disagreement_wrong = wrong_predictions_min[(wrong_predictions_min[:,2] != wrong_predictions_min[:,3]) | (wrong_predictions_min[:,2] != wrong_predictions_min[:,4]) | (wrong_predictions_min[:,2] != wrong_predictions_min[:,5])
                                   | (wrong_predictions_min[:,3] != wrong_predictions_min[:,4]) | (wrong_predictions_min[:,3] != wrong_predictions_min[:,5]) |
                                     (wrong_predictions_min[:,4] != wrong_predictions_min[:,5])]
disagreement_total = full_view_min[(full_view_min[:,2] != full_view_min[:,3]) | (full_view_min[:,2] != full_view_min[:,4]) | (full_view_min[:,2] != full_view_min[:,5])
                                   | (full_view_min[:,3] != full_view_min[:,4]) | (full_view_min[:,3] != full_view_min[:,5]) |
                                     (full_view_min[:,4] != full_view_min[:,5])]

