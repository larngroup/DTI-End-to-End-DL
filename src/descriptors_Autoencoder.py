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

# Generate Fully Connect Layers
def generate_fc(num_neurons,act_func):
    fc_layer=Dense(units=num_neurons,activation=act_func)
    return fc_layer

# Transforms data to tensors (necessary to use the functional api of keras (tensorflow based))
def generate_input(shape_size,dtype):
    data_input=Input(shape=(shape_size,),dtype=dtype)
    return data_input

#Classifier Metrics
#Sensitivty
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

# Autoencoder Model
def auto_encoder(data_train,data_test,input_length,encoder1_size,encoder2_size,encoder3_size,decoder1_size,decoder2_size,decoder3_size,fc_act_func,optimizer_func,loss_func,path,
    batch,epochs,option_validation):
    input_data=generate_input(input_length,'float32')

    encoder_1=generate_fc(encoder1_size,fc_act_func)(input_data)
    encoder_2=generate_fc(encoder2_size,fc_act_func)(encoder_1)
    encoder_3=generate_fc(encoder3_size,fc_act_func)(encoder_2)

    decoder_1=generate_fc(decoder1_size,fc_act_func)(encoder_3)
    decoder_2=generate_fc(decoder2_size,fc_act_func)(decoder_1)
    decoder_3=generate_fc(decoder3_size,'sigmoid')(decoder_2)

    model=Model(inputs=input_data,outputs=decoder_3)
    model.compile(optimizer=optimizer_func, loss=loss_func)

    #Callbacks
    early_stopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min',restore_best_weights=True)
    model_checkpoint=ModelCheckpoint(filepath=path,monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)

    if option_validation:
        model.fit(x=data_train,y=data_train,batch_size=batch,epochs=epochs,verbose=2,callbacks=[early_stopping,model_checkpoint],validation_data=(data_test,data_test))
    else:
        model.fit(x=data_train,y=data_train,batch_size=batch,epochs=epochs,verbose=2)
    
    return model


if __name__=='__main__':
    
    #Parameters
    input_length=222
    encoder1_size=128
    encoder2_size=64
    encoder3_size=32
    decoder1_size=64
    decoder2_size=128
    decoder3_size=222
    fc_act_func='relu'
    optimizer_func=Adam(lr=0.0001)
    loss_func='mse'
    batch=256
    epochs=500
    option_validation=True
    path="../Models/Autoencoder_Model.h5"

    # Load Protein Descriptors
    prot_train=np.array([i.rstrip().split(',') for i in open('../Datasets/Protein_Train_AutoEncoder.csv')])[:,1:]
    prot_test=np.array([i.rstrip().split(',') for i in open('../Datasets/Protein_Test_AutoEncoder.csv')])[:,1:]
    # Load Drug Descriptors
    drug_train=np.array([i.rstrip().split(',') for i in open('../Datasets/Drug_Train_AutoEncoder.csv')])[:,1:]
    drug_test=np.array([i.rstrip().split(',') for i in open('../Datasets/Drug_Test_AutoEncoder.csv')])[:,1:]
    
    # Build Training and Testing Dataset
    data_train=np.hstack([prot_train.astype('float32'),drug_train.astype('float32')])
    data_test=np.hstack([prot_test.astype('float32'),drug_test.astype('float32')])
    
    # Scaling
    scaler=MinMaxScaler().fit(data_train)
    data_train=scaler.transform(data_train)
    data_test=scaler.transform(data_test)

    # Autoencoder Model
    model=auto_encoder(data_train,data_test,input_length,encoder1_size,encoder2_size,encoder3_size,decoder1_size,decoder2_size,decoder3_size,fc_act_func,optimizer_func,loss_func,path,
    batch,epochs,True)
