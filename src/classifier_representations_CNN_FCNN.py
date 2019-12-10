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
        
# Sigmoid Results to Binary    
def sigmoid_to_binary(predicted_labels):
    binary_labels=[]
    for i in predicted_labels:
        if i>0.5:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

#def weighted_binary_crossentropy(y_true, y_pred):
#    b_ce = K.binary_crossentropy(y_true, y_pred)
#    weight_vector = y_true * 0.36 + (1. - y_true) * 0.64
#    weighted_b_ce = weight_vector * b_ce
#
#        # Return the mean error
#    return K.mean(weighted_b_ce)


# Create One Hot Layer
def OneHot(input_dim=None, input_length=None):
    def one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),num_classes=num_classes)

    ## Lambda Layer allows to create a special layer that represents the result of some kind of operation over the data
    return Lambda(one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))

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

# Generate Covolutional Layers
def generate_cov1D(num_filters,filter_window,stride,padding_method,act_func):
    cov_layer=Conv1D(filters=num_filters,kernel_size=filter_window,strides=stride,padding=padding_method,activation=act_func)
    return cov_layer

# Generate Fully Connect Layers
def generate_fc(num_neurons,act_func):
    fc_layer=Dense(units=num_neurons,activation=act_func)
    return fc_layer

# Generate embedding layers (needs to be the first layer)
def generate_embedding(input_dim,output_dim,input_len):
    embedding=Embedding(input_dim=input_dim+1,output_dim=output_dim,input_length=input_len)
    return embedding

# Transforms data to tensors (necessary to use the functional api of keras (tensorflow based))
def generate_input(shape_size,dtype):
    data_input=Input(shape=(shape_size,),dtype=dtype)
    return data_input

# Generate Max or Average Pooling Layers
def generate_pooling(type,pool_size):
    if type=='max':
        max_pool=MaxPooling1D(pool_size=pool_size,padding='valid')
        return max_pool
    elif type=='average':
        average_pool=AveragePooling1D(pool_size=pool_size,padding='valid')
        return average_pool

# List of optimizers with different learning rates
def generate_optimizers(lr_rate):
    optimizer=[Adam(lr=lr_rate),SGD(lr=lr_rate),RMSprop(lr=lr_rate),Adamax(lr=lr_rate),Nadam(lr=lr_rate)]
    return optimizer

#Classifier Metrics
#Sensitivity
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
        metrics.append('AUPRC: '+str(auprc))
    return metrics

# Classifier
def cnn_classifier(prot_data,smile_data,labels,prot_val,smile_val,labels_val,prot_seq_len,smile_len,encoding_type,prot_dict_size,
    embedding_size,smile_dict_size,number_cov_layers,num_filters,prot_filter_1,prot_filter_2,prot_filter_3,prot_filter_4,prot_filter_5,
    act_func_conv,smile_filter_1,smile_filter_2,smile_filter_3,smile_filter_4,smile_filter_5,number_fc_layers,fc_neurons_1,fc_neurons_2,
    fc_neurons_3,fc_neurons_4,fc_act_func,drop_rate,output_act,optimizer_func,loss_func,metric_type,batch,epochs,option_validation,path):


    # Inputs tensor based
    protein_input=generate_input(prot_seq_len,'int32')
    smile_input=generate_input(smile_len,'int32')
    
    # Encoding Type
    if encoding_type=='embedding':
        #Embedding Layers: it allows to map integers into useful continuous vectors with a certain dimension (embedding size)
        protein_embedding=generate_embedding(prot_dict_size,embedding_size,prot_seq_len)(protein_input)
        smile_embedding=generate_embedding(smile_dict_size,embedding_size,smile_len)(smile_input)
    elif encoding_type=='one_hot':
        protein_embedding=OneHot(input_dim=prot_dict_size, input_length=prot_seq_len)(protein_input)
        smile_embedding=OneHot(input_dim=smile_dict_size,input_length=smile_len)(smile_input)


    # Convolutional Layers
    # Proteins
    if number_cov_layers==1:
        protein_cnn=generate_cov1D(num_filters,prot_filter_1,1,'valid',act_func_conv)(protein_embedding)
    elif number_cov_layers==2:
        protein_cnn=generate_cov1D(num_filters,prot_filter_1,1,'valid',act_func_conv)(protein_embedding)
        protein_cnn=generate_cov1D(num_filters*2,prot_filter_2,1,'valid',act_func_conv)(protein_cnn)
    elif number_cov_layers==3:
        protein_cnn=generate_cov1D(num_filters,prot_filter_1,1,'valid',act_func_conv)(protein_embedding)
        protein_cnn=generate_cov1D(num_filters*2,prot_filter_2,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*3,prot_filter_3,1,'valid',act_func_conv)(protein_cnn)
    elif number_cov_layers==4:
        protein_cnn=generate_cov1D(num_filters,prot_filter_1,1,'valid',act_func_conv)(protein_embedding)
        protein_cnn=generate_cov1D(num_filters*2,prot_filter_2,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*3,prot_filter_3,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*4,prot_filter_4,1,'valid',act_func_conv)(protein_cnn)
    elif number_cov_layers==5:
        protein_cnn=generate_cov1D(num_filters,prot_filter_1,1,'valid',act_func_conv)(protein_embedding)
        protein_cnn=generate_cov1D(num_filters*2,prot_filter_2,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*3,prot_filter_3,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*4,prot_filter_4,1,'valid',act_func_conv)(protein_cnn)
        protein_cnn=generate_cov1D(num_filters*5,prot_filter_5,1,'valid',act_func_conv)(protein_cnn)

    # Smiles
    if number_cov_layers==1:
        smile_cnn=generate_cov1D(num_filters,smile_filter_1,1,'valid',act_func_conv)(smile_embedding)
    elif number_cov_layers==2:
        smile_cnn=generate_cov1D(num_filters,smile_filter_1,1,'valid',act_func_conv)(smile_embedding)
        smile_cnn=generate_cov1D(num_filters*2,smile_filter_2,1,'valid',act_func_conv)(smile_cnn)
    elif number_cov_layers==3:
        smile_cnn=generate_cov1D(num_filters,smile_filter_1,1,'valid',act_func_conv)(smile_embedding)
        smile_cnn=generate_cov1D(num_filters*2,smile_filter_2,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*3,smile_filter_3,1,'valid',act_func_conv)(smile_cnn)
    elif number_cov_layers==4:
        smile_cnn=generate_cov1D(num_filters,smile_filter_1,1,'valid',act_func_conv)(smile_embedding)
        smile_cnn=generate_cov1D(num_filters*2,smile_filter_2,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*3,smile_filter_3,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*4,smile_filter_4,1,'valid',act_func_conv)(smile_cnn)
    elif number_cov_layers==5:
        smile_cnn=generate_cov1D(num_filters,smile_filter_1,1,'valid',act_func_conv)(smile_embedding)
        smile_cnn=generate_cov1D(num_filters*2,smile_filter_2,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*3,smile_filter_3,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*4,smile_filter_4,1,'valid',act_func_conv)(smile_cnn)
        smile_cnn=generate_cov1D(num_filters*5,smile_filter_5,1,'valid',act_func_conv)(smile_cnn)


        
    # Pooling Layers
    # Proteins
    protein_pool=GlobalMaxPooling1D()(protein_cnn)
    # Smiles
    smile_pool=GlobalMaxPooling1D()(smile_cnn)
    
    # Merge Features Representations
    features=[protein_pool,smile_pool]
    features=concatenate(features)
    
    #Fully Connected
    if number_fc_layers==1:
        fc_layer=generate_fc(fc_neurons_1,fc_act_func)(features)

    elif number_fc_layers==2:
        fc_layer=generate_fc(fc_neurons_1,fc_act_func)(features)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_2,fc_act_func)(fc_layer)

    elif number_fc_layers==3:
        fc_layer=generate_fc(fc_neurons_1,fc_act_func)(features)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_2,fc_act_func)(fc_layer)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_3,fc_act_func)(fc_layer)

    elif number_fc_layers==4:
        fc_layer=generate_fc(fc_neurons_1,fc_act_func)(features)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_2,fc_act_func)(fc_layer)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_3,fc_act_func)(fc_layer)
        fc_layer=Dropout(rate=drop_rate)(fc_layer)
        fc_layer=generate_fc(fc_neurons_4,fc_act_func)(fc_layer)

    
    # Output Layer
    output=generate_fc(1,output_act)(fc_layer)
    
    # Model
    model = Model(inputs=[protein_input,smile_input], outputs=output)
    
    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=metric_type)

    #Callbacks
    early_stopping=EarlyStopping(monitor='val_f1_score', min_delta=0, patience=50, verbose=0, mode='max',restore_best_weights=True)
    model_checkpoint=ModelCheckpoint(filepath=path,monitor='val_f1_score', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
   

    if option_validation:
        model.fit(x=[prot_data,smile_data],y=labels,batch_size=batch,epochs=epochs,verbose=2,callbacks=[early_stopping,model_checkpoint],validation_data=([prot_val,smile_val],labels_val),class_weight = {0: 0.36, 1: 0.64})
    else:
        model.fit(x=[prot_data,smile_data],y=labels,batch_size=batch,epochs=epochs,verbose=0)
    
    return model



def grid_search(prot_train,smile_train,labels_train,prot_test,smile_test,labels_test,number_cov_layers,number_fc_layers,prot_seq_len,smile_len,prot_dict_size,smile_dict_size,encoding_type,embedding_size,
    num_filters,drop_rate,batch,learning_rate,prot_filter_1_window,prot_filter_2_window,prot_filter_3_window,prot_filter_4_window,prot_filter_5_window,smile_filter_1_window,smile_filter_2_window,smile_filter_3_window,
    smile_filter_4_window,smile_filter_5_window,fc_1_size,fc_2_size,fc_3_size,fc_4_size,act_func_conv,fc_act_func,epochs,loss_func,output_act,metric_type):
    for n_filter in num_filters:
        for d_rate in drop_rate:
            for l_rate in learning_rate:
                for smile_filter_1 in smile_filter_1_window:
                    for smile_filter_2 in smile_filter_2_window:
                        for smile_filter_3 in smile_filter_3_window:
                            for prot_filter_1 in prot_filter_1_window:
                                for prot_filter_2 in prot_filter_2_window:
                                    for prot_filter_3 in prot_filter_3_window:
                                        for fc_neurons_1 in fc_1_size:
                                            for fc_neurons_2 in fc_2_size:
                                                for fc_neurons_3 in fc_3_size:
                                                    file_name=[str(n_filter)+'_'+str(d_rate)+'_'+str(l_rate)+'_'+str(smile_filter_1)+'_'+str(smile_filter_2)+'_'+str(smile_filter_3)+'_'+str(prot_filter_1)+'_'+
                                                            str(prot_filter_2)+'_'+str(prot_filter_3)+'_'+str(fc_neurons_1)+'_'+str(fc_neurons_2)+'_'+str(fc_neurons_3)][0]
                                                    print(file_name)
                                                    model=cnn_classifier(prot_train,smile_train,labels_train,prot_test,smile_test,labels_test,prot_seq_len,smile_len,encoding_type,
                                                        prot_dict_size,0,smile_dict_size,number_cov_layers,n_filter,prot_filter_1,prot_filter_2,prot_filter_3,0,0,
                                                        act_func_conv,smile_filter_1,smile_filter_2,smile_filter_3,0,0,number_fc_layers,fc_neurons_1,fc_neurons_2,fc_neurons_3,0,
                                                        fc_act_func,d_rate,output_act,Adam(lr=l_rate),loss_func,metric_type,batch,epochs,True,"./Models_CNN_FCNN/"+file_name+".h5")
                                                    predicted_labels=model.predict([prot_test,smile_test])
                                                    binary_labels=sigmoid_to_binary(predicted_labels)
                                                    cm=confusion_matrix(labels_test,np.array(binary_labels))
                                                    #print(cm)
                                                    metric_values=metrics_function(True,True,True,True,False,False,binary_labels,predicted_labels,labels_test,cm)
                                                    #print(metric_values)
                                                    save_func('../Results_CNN_FCNN.csv',[n_filter,d_rate,batch,l_rate,smile_filter_1,smile_filter_2,smile_filter_3,prot_filter_1,prot_filter_2,
                                                    prot_filter_3,fc_neurons_1,fc_neurons_2,fc_neurons_3,epochs,loss_func,output_act,act_func_conv,fc_act_func,
                                                    metric_values[0].strip('Sensitivity:'),metric_values[1].strip('Specificity:'),metric_values[2].strip('F1_Score:'),
                                                    metric_values[3].strip('Accuracy:')])                               
                                                                                                    


                                                                            

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




   # _________________Parameters______________________
   # Protein and Smile Length
   prot_seq_len=1205
   smile_len=90
   ## Protein and Smile Dictionary Length
   prot_dict_size=len(prot_dictionary)
   smile_dict_size=len(smile_dictionary)
   ## Number of filters of the convolutional layers
   num_filters=[48,32,64,96,128]
   ## Size of the embedding vector 
   embedding_size=[0]
   ## Size of the filter windows
   prot_filter_window=[2,3,4,5,6]
   smile_filter_window=[2,3,4,5,6]
   ## Activation functions
   act_func_conv='relu'
   fc_act_func='relu'
   ## Drop Rate
   drop_rate=[0.1,0.3,0.5,0.7]
   ## Batch Size
   batch=256
   ## Number of Epochs
   epochs=500
   ## FC Layer size
   fc_size=[32,64,128,256,512,1024]
   ## Number of Convolution Layers
   number_cov_layers=3
   ## Number of Fully Connected Layers
   number_fc_layers=3
   ## Learning Rate
   learning_rate=[0.0001,0.001,0.01,0.1]
   ## Loss Functions
   loss_func='binary_crossentropy'
   ## Output Activation Function
   output_act='sigmoid'
   ## Metrics
   metric_type=['accuracy',sensitivity,specificity,f1_score]
   ## Enconding type
   encoding_type='one_hot'


   grid_search(prot_train_data,smile_train_data,labels_train,prot_test_data,smile_test_data,labels_test,number_cov_layers,number_fc_layers,prot_seq_len,smile_len,prot_dict_size,smile_dict_size,encoding_type,embedding_size,
   num_filters,drop_rate,batch,learning_rate,prot_filter_window,prot_filter_window,prot_filter_window,[0],[0],smile_filter_window,smile_filter_window,smile_filter_window,
   [0],[0],fc_size,fc_size,fc_size,[0],act_func_conv,fc_act_func,epochs,loss_func,output_act,metric_type)





    
    














