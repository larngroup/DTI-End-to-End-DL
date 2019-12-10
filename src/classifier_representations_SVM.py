import numpy.core.multiarray as multiarray
import json
import itertools
import multiprocessing
import pickle
from sklearn import svm
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
from tensorflow.python.ops import math_ops
from keras import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from keras.layers.advanced_activations import *
from keras.optimizers import *
from keras.callbacks import *
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import StratifiedKFold as SKF

os.environ["CUDA_VISIBLE_DEVICES"]="0"
session = tf.Session()
K.set_session(session)

# Data Conversion using Dictionary
def data_conversion(dataset,dictionary,max_length):
    matrix=np.zeros([len(dataset),max_length])
    for i in range(len(dataset)):
        dataset[i][1]=list(dataset[i][1])
        for j in range(len(dataset[i][1])):
            for k in dictionary:
                if dataset[i][1][j]==k:
                    dataset[i][1][j]=dictionary.get(k)
        if len(dataset[i][1])<max_length:
            matrix[i,0:len(dataset[i][1])]=dataset[i][1]
        else:
            matrix[i,0:max_length]=dataset[i][1][0:max_length]
    return matrix.astype('int32')

#CNN Model Metrics
# Sensitivity
def sensitivity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TP,TP+FN)
    return metric
# Specificity
def specificity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    return metric
# F1-Score
def f1_score(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    precision = tf.divide(TP,TP + FP)
    sensitivity = tf.divide(TP,TP+FN)
    metric = tf.divide(tf.multiply(2*precision,sensitivity),precision + sensitivity)
    return metric

# Metrics Function: Sensitivity, Specificity, F1-Score, Accuracy and AUC
def metrics_function(sensitivity,specificity,f1,accuracy,auc_value,auprc_value,binary_labels,predicted_labels,labels_test,confusion_matrix):
    sensitivity_value=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
    specificity_value= confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    precision=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    f1_value=2*(precision*sensitivity_value)/(precision+sensitivity_value)
    accuracy=accuracy_score(labels_test,np.array(binary_labels))
    auc=roc_auc_score(labels_test,predicted_labels)
    auprc=average_precision_score(labels_test,predicted_labels)
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
        metrics.append('AUPRC:'+str(auprc))
    return metrics

# Get CNN Model Deep Representations
def get_cnn_layers(model,data,prot,smile):
    result=[]
    if prot:
        m2=Model(inputs=model.layers[0].input, outputs=model.layers[10].output)
        result=m2.predict(data)
    elif smile:
        m2=Model(inputs=model.layers[1].input, outputs=model.layers[11].output)
        result=m2.predict(data)
    return result

# SVM Model Grid Search
def SVM_gridsearch(parameters,data_train,labels_train,number_splits,num_threads):
    svm_clf=svm.SVC(gamma="scale",probability=True)
    # multiprocessing.cpu_count()
    clf = GSCV(svm_clf, parameters, cv=SKF(n_splits=number_splits),verbose=2,n_jobs=num_threads)
    clf.fit(data_train,labels_train)
    return clf

if __name__ == '__main__':    
   ## Load Protein Sequences
   prot_train=[i.rstrip().split(',') for i in open('../Datasets/Protein_Train_Dataset.csv')]
   prot_test=[i.rstrip().split(',') for i in open('../Datasets/Protein_Test_Dataset.csv')]

   ## Load SMILE Strings
   drug_train=[i.rstrip().split(',') for i in open('../Datasets/Smile_Train_Dataset.csv')]
   drug_test=[i.rstrip().split(',') for i in open('../Datasets/Smile_Test_Dataset.csv')]

   ## Load Protein & Smile Dictionaries
   prot_dictionary=json.load(open('../Dictionaries/aa_properties_dictionary.txt'))
   smile_dictionary=json.load(open('../Dictionaries/smile_dictionary.txt'))

   
   #Sequences and SMILES to integers
   prot_train_data=data_conversion(prot_train,prot_dictionary,1205)
   prot_test_data=data_conversion(prot_test,prot_dictionary,1205)

   smile_train_data=data_conversion(drug_train,smile_dictionary,90)
   smile_test_data=data_conversion(drug_test,smile_dictionary,90)

   ## Labels
   labels_train=np.load('../Labels/labels_train.npy')
   labels_test=np.load('../Labels/labels_test.npy')
   
   
   ## Load CNN Model
   cnn_model=load_model('../Models/CNN_Model.h5',custom_objects={'sensitivity':sensitivity,'specificity':specificity,'f1_score':f1_score})

   ## CNN Deep Representations
   cnn_prot_train=get_cnn_layers(cnn_model,prot_train_data,True,False)
   cnn_prot_test=get_cnn_layers(cnn_model,prot_test_data,True,False)
   cnn_smile_train=get_cnn_layers(cnn_model,smile_train_data,False,True)
   cnn_smile_test=get_cnn_layers(cnn_model,smile_test_data,False,True)
   
   ## Training Dataset and Testing Dataset based on deep representations
   data_train=np.hstack([cnn_prot_train.astype('float32'),cnn_smile_train.astype('float32')])
   data_test=np.hstack([cnn_prot_test.astype('float32'),cnn_smile_test.astype('float32')])
   
   # Parameters
   parameters = {'kernel':('linear', 'rbf','sigmoid','poly'), 'C':[0.001, 0.01, 0.1, 1, 10],'degree':[2,3,4,5]}
   
   # Grid Search
   svm_model=SVM_gridsearch(parameters, data_train, labels_train,5,10)
   
   # Save Model
   pickle.dump(svm_model,open("../Models/SVM_Model_Representations.py",'wb'))
#   svm_model=pickle.load(open('./Models/SVM_Model_Representations.py','rb'))
   
   # Model Parameters
   params=svm_model.get_params()
   print(params)
   
   # Predicted Labels
   pred_labels=svm_model.predict(data_test)
   
   # Confusion Matrix
   cm=confusion_matrix(labels_test,pred_labels)
   print(cm)
   
   #Metrics
   metric_values=metrics_function(True,True,True,True,False,False,pred_labels,svm_model.predict_proba(data_test)[:,1],labels_test,cm)
   print(metric_values)
   
   
