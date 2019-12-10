import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

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


# Random Forest Classifier
def RF_Classifier(data_train,labels_train,num_estimators,max_features,oob_score,n_jobs):
    random_forest=RF(n_estimators=num_estimators,max_features=max_features,oob_score=oob_score,n_jobs=n_jobs)
    random_forest.fit(data_train,labels_train)
    return random_forest


if __name__ == '__main__':
    # Load Training Dataset
    data_train=np.array([i.rstrip().split(',')[3:] for i in open('../Datasets/Descriptors_Train_Dataset.csv')]).astype('float64')
    
    # Load Testing Dataset
    data_test=np.array([i.rstrip().split(',')[3:] for i in open('../Datasets/Descriptors_Test_Dataset.csv')]).astype('float64')
 
    # Load Labels
    labels_train=np.load('../Labels/labels_train.npy')
    labels_test=np.load('../Labels/labels_test.npy')
    
    # Normalization 
    data_train=normalize(data_train)
    data_test=normalize(data_test)
    
    # Random Forest Model
    rf_clf=RF_Classifier(data_train,labels_train,150,100,True,-1)
    
    # Save RF Model
    pickle.dump(rf_clf,open("../Models/RF_Model.py",'wb'))
    #rf_clf=pickle.load(open('./Models/RF_Model.py','rb'))
    
    # Predicted Labels
    pred_labels=rf_clf.predict(data_test)
    
    # Confusion Matrix
    cm=confusion_matrix(labels_test,pred_labels)
    print(cm)
    
    # Metrics 
    metric_values=metrics_function(True,True,True,True,False,False,pred_labels,rf_clf.predict_proba(data_test)[:,1],labels_test,cm)
    print(metric_values)
    
    
    
    