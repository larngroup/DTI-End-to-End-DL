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

# Function to save best result
def save_func(file_path,values):
    file=[i.rstrip().split(',') for i in open(file_path).readlines()]
    file.append(values)
    file=pd.DataFrame(file)
    file.to_csv(file_path,header=None,index=None)

# Transforms data to tensors (necessary to use the functional api of keras (tensorflow based))
def generate_input(shape_size,dtype):
    data_input=Input(shape=(shape_size,),dtype=dtype)
    return data_input

# Generate Fully Connect Layers    
def generate_fc(num_neurons,act_func):
    fc_layer=Dense(units=num_neurons,activation=act_func)
    return fc_layer

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


# Sigmoid Results to Binary     
def sigmoid_to_binary(predicted_labels):
    binary_labels=[]
    for i in predicted_labels:
        if i>0.5:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

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
        metrics.append('AUPRC: '+str(auprc))
    return metrics

# FCNN Model
def cnn_autoencoder_fcnn(data_train,data_test,input_length,prot_data,smile_data,labels,prot_val,smile_val,labels_val,prot_seq_len,smile_len,
    fc_neurons_1,fc_neurons_2,fc_neurons_3,drop_rate,fc_act_func,output_act,optimizer_func,loss_func,path,
    batch,epochs,option_validation,metric_type):
    ## Input
    protein_input=generate_input(prot_seq_len,'float32')
    smile_input=generate_input(smile_len,'float32')
    data=generate_input(input_length,'float32')

    # Concatenation
    features=[protein_input,smile_input,data]
    features=concatenate(features)
    
    ## Fully Connected Neural Network
    fc_layer=generate_fc(fc_neurons_1,fc_act_func)(features)
    fc_layer=Dropout(rate=drop_rate)(fc_layer)
    fc_layer=generate_fc(fc_neurons_2,fc_act_func)(fc_layer)
    fc_layer=Dropout(rate=drop_rate)(fc_layer)
    fc_layer=generate_fc(fc_neurons_3,fc_act_func)(fc_layer)
    
    ## Output Layer
    output=generate_fc(1,output_act)(fc_layer)
    
    ## Model
    model=Model(inputs=[protein_input,smile_input,data],outputs=output)
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=metric_type)

    #Callbacks
    early_stopping=EarlyStopping(monitor='val_f1_score', min_delta=0, patience=50, verbose=0, mode='max',restore_best_weights=True)
    model_checkpoint=ModelCheckpoint(filepath=path,monitor='val_f1_score', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)

    if option_validation:
        model.fit(x=[prot_data,smile_data,data_train],y=labels,batch_size=batch,epochs=epochs,verbose=2,callbacks=[early_stopping,model_checkpoint],validation_data=([prot_val,smile_val,data_test],labels_val),class_weight = {0: 0.36, 1: 0.64})
    else:
        model.fit(x=[prot_data,smile_data,data_train],y=labels,batch_size=batch,epochs=epochs,verbose=0)
    
    return model

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

# Get Autoencoder Model Deep Representations
def get_autoencoder_layers(model,data):
    m2=Model(inputs=model.layers[0].input, outputs=model.layers[3].output)
    result=m2.predict(data)
    return result



# Grid Search Based on Early Stopping and Model Checkpoint with F1-score as the evaluation metric
def grid_search(data_train,data_test,input_length,prot_data,smile_data,labels,prot_val,smile_val,labels_val,prot_seq_len,smile_len,
    fc_1_size,fc_2_size,fc_3_size,drop_rate,learning_rate,fc_act_func,output_act,loss_func,
    batch,epochs,option_validation,metric_type):
    for d_rate in drop_rate:
        for l_rate in learning_rate:
            for fc_neurons_1 in fc_1_size:
                for fc_neurons_2 in fc_2_size:
                    for fc_neurons_3 in fc_3_size:
                        file_name=[str(d_rate)+'_'+str(l_rate)+'_'+str(fc_neurons_1)+'_'+str(fc_neurons_2)+'_'+str(fc_neurons_3)][0]
                        print(file_name)
                        model=cnn_autoencoder_fcnn(data_train,data_test,input_length,prot_data,smile_data,labels,prot_val,smile_val,labels_val,prot_seq_len,smile_len,
                            fc_neurons_1,fc_neurons_2,fc_neurons_3,d_rate,fc_act_func,output_act,Adam(lr=l_rate),loss_func,"./Models_FCNN_Combined_Model/"+file_name+".h5",
                            batch,epochs,option_validation,metric_type)
                        #loss,acc,sensi,speci,f1=model.evaluate(([cnn_prot_test,cnn_smile_test,autoencoder_test]),labels_test,verbose=1)
                        predicted_labels=model.predict([cnn_prot_test,cnn_smile_test,autoencoder_test])
                        binary_labels=sigmoid_to_binary(predicted_labels)
                        cm=confusion_matrix(labels_test,np.array(binary_labels))
                        #print(cm)
                        metric_values=metrics_function(True,True,True,True,False,False,binary_labels,predicted_labels,labels_test,cm)
                        #print(metric_values)
                        save_func('../Results_FCNN_Combined_Model.csv',[d_rate,l_rate,fc_neurons_1,fc_neurons_2,fc_neurons_3,
                            metric_values[0].strip('Sensitivity:'),metric_values[1].strip('Specificity:'),metric_values[2].strip('F1_Score:'),
                            metric_values[3].strip('Accuracy:')])
  

                                                    


if __name__ == '__main__':    
#################### CNN Model########################
   ## Load Protein Sequences
   prot_train=[i.rstrip().split(',') for i in open('../Datasets/Protein_Train_Dataset.csv')]
   prot_test=[i.rstrip().split(',') for i in open('../Datasets/Protein_Test_Dataset_AE.csv')]

   ## Load SMILE Strings
   drug_train=[i.rstrip().split(',') for i in open('../Datasets/Smile_Train_Dataset.csv')]
   drug_test=[i.rstrip().split(',') for i in open('../Datasets/Smile_Test_Dataset_AE.csv')]

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
   labels_test=np.load('../Labels/labels_test_AE.npy')
   
   
   ## Load CNN Model
   cnn_model=load_model('../Models/CNN_Model.h5',custom_objects={'sensitivity':sensitivity,'specificity':specificity,'f1_score':f1_score})

#################### Autoencoder Model########################
   # Load Protein Descriptors
   prot_train_auto=np.array([i.rstrip().split(',') for i in open('../Datasets/Protein_Train_AutoEncoder.csv')])[:,1:]
   prot_test_auto=np.array([i.rstrip().split(',') for i in open('../Datasets/Protein_Test_AutoEncoder.csv')])[:,1:]
   # Load Drug Descriptors
   drug_train_auto=np.array([i.rstrip().split(',') for i in open('../Datasets/Drug_Train_AutoEncoder.csv')])[:,1:]
   drug_test_auto=np.array([i.rstrip().split(',') for i in open('../Datasets/Drug_Test_AutoEncoder.csv')])[:,1:]
   
   # Build Training and Testing Dataset
   data_train_auto=np.hstack([prot_train_auto.astype('float32'),drug_train_auto.astype('float32')])
   data_test_auto=np.hstack([prot_test_auto.astype('float32'),drug_test_auto.astype('float32')])

   # Scaling
   scaler=MinMaxScaler().fit(data_train_auto)
   data_train_auto=scaler.transform(data_train_auto)
   data_test_auto=scaler.transform(data_test_auto)

   # Load Autoencoder Model
   model_autoenconder=load_model('../Models/Autoencoder_Model.h5')
   
#################### Combined Model########################  
    
   # Deep Representations
   # CNN Deep Representations
   cnn_prot_train=get_cnn_layers(cnn_model,prot_train_data,True,False)
   cnn_prot_test=get_cnn_layers(cnn_model,prot_test_data,True,False)
   cnn_smile_train=get_cnn_layers(cnn_model,smile_train_data,False,True)
   cnn_smile_test=get_cnn_layers(cnn_model,smile_test_data,False,True)
   # Autoencoder Deep Representations
   autoencoder_train=get_autoencoder_layers(model_autoenconder,data_train_auto)
   autoencoder_test=get_autoencoder_layers(model_autoenconder,data_test_auto)
    

    
   metric_type=['accuracy',sensitivity,specificity,f1_score]
   smile_len=384
   prot_seq_len=384
   input_length=32
   fc_1_size=[128,256,512,1024]
   fc_2_size=[128,256,512,1024]
   fc_3_size=[128,256,512,1024]
   drop_rate=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
   output_act='sigmoid'
   fc_act_func='relu'
   learning_rate=[0.001,0.0001]
   loss_func='binary_crossentropy'
   batch=256
   epochs=500
   option_validation=True
   grid_search(autoencoder_train,autoencoder_test,input_length,cnn_prot_train,cnn_smile_train,labels_train,
               cnn_prot_test,cnn_smile_test,labels_test,prot_seq_len,smile_len,fc_1_size,fc_2_size,fc_3_size,
               drop_rate,learning_rate,fc_act_func,output_act,loss_func,batch,epochs,option_validation,metric_type)



