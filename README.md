# Drug-Target Interaction Prediction: End-to-End Deep Learning Approach
<p align="justify">We propose an end-to-end deep learning pipeline, capable of automatically identify important sequencial regions from 1D raw data, protein sequences and SMILES (Simplified Molecular Input Line Entry System) strings, and extract deep representations using Convolutional Neural Networks (CNN). A Fully Connected Neural Network is applied as a binary classifier, using as input the representations obtained from the CNNs.
Additionally, we compare the proposed model with traditional machine learning approaches using both global descriptors and deep representations, a deep learning architecture based on global descriptors and also a combined deep learning model using a mix of descriptors, encoded into deep representations, and protein and compounds deep representations.</p>
  
## Two Parallel Convolution Neural Networks + Fully Connected Neural Network
<img src="/figures/CNN_FCNN_Model.png"/>

## Two Parallel Convolution Neural Networks (Pre-Trained) + Autoencoder (Pre-Trained) + Fully Connected Neural Network
<img src="/figures/CNN_Autoencoder_FCNN_Model.png"/>

## Grid Search
<img src="/figures/Gridsearch.png"/>

## Models:
- **Two Parallel Convolution Neural Networks + Fully Connected Neural Network** (Deep Representations)
- **Fully Connected Neural Network** (Descriptors)
- **Two Parallel Convolution Neural Networks (Pre-Trained) + Autoencoder (Pre-Trained) + Fully Connected Neural Network** (Deep Representations)
- **Random Forest** (Descriptors & Deep Representations)
- **Support Vector Machine** (Descriptors & Deep Representations)
- **Autoencoder** (Particular Group of Descriptors)

## Datasets
- **Descriptors_Test_Dataset**: SVM Descriptors, RF Descriptors, FCNN Descriptors
- **Descriptors_Train_Dataset**: SVM Descriptors, RF Descriptors, FCNN Descriptors
- **Drug_Test_AutoEncoder**: Autoencoder, CNN+Autoencoder+FCNN
- **Drug_Train_AutoEncoder**: Autoencoder, CNN+Autoencoder+FCNN
- **Protein_Test_AutoEncoder**: Autoencoder, CNN+Autoencoder+FCNN
- **Protein_Train_AutoEncoder**: Autoencoder, CNN+Autoencoder+FCNN
- **Protein_Test_Dataset**: CNN+FCNN, SVM Representations, RF Representations
- **Protein_Train_Dataset**: CNN+FCNN, SVM Representations, RF Representations
- **Smile_Test_Dataset**: CNN+FCNN, SVM Representations, RF Representations
- **Smile_Train_Dataset**: CNN+FCNN, SVM Representations, RF Representations
- **Protein_Test_Dataset_AE**: CNN+Autoencoder+FCNN
- **Smile_Test_Dataset_AE**: CNN+Autoencoder+FCNN

## Labels
- **Labels Train**: all models
- **Labels Test**: all models except CNN+Autoencoder+FCNN
- **Labels Test AE**: CNN+Autoencoder+FCNN (1 less sample)

## Dictionaries
- **aa_properties_dictionary**: protein encoding dictionary based on AA physicochemical properties
- **protein_dictionary**: AA char-integer dictionary (not used)
- **smile_dictionary**: SMILES char-integer dictionary

## Requirements:
- Python 3.6.6
- Tensorflow 1.x
- Keras 2.x
- Numpy 
- Pandas
- Scikit-learn
- Json
- Pickle
- OS
