# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:38:36 2019

@author: Nelson
"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import StratifiedKFold as SKF
import multiprocessing


# Metrics Function: Sensitivity, Specificity, F1-Score, Accuracy and AUC
def metrics_function(sensitivity,specificity,f1,accuracy,auc_value,auprc_value,predicted_labels,pred_prob,labels_test,confusion_matrix):
    sensitivity_value=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
    specificity_value= confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    precision=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    f1_value=2*(precision*sensitivity_value)/(precision+sensitivity_value)
    accuracy=accuracy_score(labels_test,predicted_labels)
    auc=roc_auc_score(labels_test,pred_prob)
    auprc=average_precision_score(labels_test,pred_prob)
    metrics=[]
    if sensitivity:
        metrics.append('Sensitivity:'+str(sensitivity_value))
    if specificity:
        metrics.append('Specificity:'+str(specificity_value))
    if f1:
        metrics.append('F1_Score:'+str(f1_value))
    if accuracy:
        metrics.append('Accuracy:'+str(accuracy))
    if auc_value:
        metrics.append('AUC:'+str(auc))
    if auprc_value:
        metrics.append('AUPRC: '+str(auprc))
    return metrics    
  

# SVM Grid Search based on Stratified K Fold Cross Validation
def SVM_gridsearch(parameters,data_train,labels_train,number_splits,num_threads):
    svm_clf=svm.SVC(gamma="scale",probability=True)
    # multiprocessing.cpu_count()
    clf = GSCV(svm_clf, parameters, cv=SKF(n_splits=number_splits),verbose=2,n_jobs=num_threads)
    clf.fit(data_train,labels_train)
    return clf

## SVM Model
if __name__=='__main__':
    
    # Parameters
    parameters = {'kernel':('linear', 'rbf','sigmoid','poly'), 'C':[0.001, 0.01, 0.1, 1, 10],'degree':[2,3,4,5]}
    
    # Load Training Dataset
    data_train=np.array([i.rstrip().split(',')[3:] for i in open('../Datasets/Descriptors_Train_Dataset.csv')]).astype('float64')
    
    # Load Testing Dataset
    data_test=np.array([i.rstrip().split(',')[3:] for i in open('../Datasets/Descriptors_Test_Dataset.csv')]).astype('float64')
    
    # Load Labels
    labels_train=np.load('../Labels/labels_train.npy')
    labels_test=np.load('../Labels/labels_test.npy')
    
    # Scaling
    scaler=MinMaxScaler().fit(data_train)
    data_train=scaler.transform(data_train)
    data_test=scaler.transform(data_test)
    
    # Grid Search
    model=SVM_gridsearch(parameters,data_train,labels_train,5,10)
    
    # Save Model
    pickle.dump(model,open('../Models/SVM_Model.py','wb'))
#    model=pickle.load(open('./Models/SVM_Model.py','rb'))
    
    # Model Parameters
    params=model.get_params()
    print(params)
    
    # Predicted Labels
    pred_labels=model.predict(data_test)
    
    # Confusion Matrix
    cm=confusion_matrix(labels_test,pred_labels)
    print(cm)
    
    # Metrics
    metric_values=metrics_function(True,True,True,True,False,False,pred_labels,model.predict_proba(data_test)[:,1],labels_test,cm)
    print(metric_values)
    
    
    