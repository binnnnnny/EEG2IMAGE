""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

#%%
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

import models 
from preprocess import get_data
import wandb
from wandb.keras import WandbCallback
# from keras.utils.vis_utils import plot_model


#%%
def draw_learning_curves(history):
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.show()

def draw_confusion_matrix(cf_matrix, results_path, classes_labels):
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                  display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_xticklabels(display_labels, rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(results_path + '/confusion_matrix.png')
    plt.show()

def draw_performance_barChart(metric, label):
    fig, ax = plt.subplots()
    x = np.arange(len(metric))  
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_title('Model ' + label)
    ax.set_xticks(x)

    ax.set_ylim([0, 1])
    plt.show()
    
    
#%% Training 
def train(dataset_conf, train_conf, results_path):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)        

    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    
    data_path = dataset_conf.get('data_path')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    from_logits = train_conf.get('from_logits')

    print('\nTraining started')
    wandb.init(project='Rxde',name='ATCNet_ver2')
    bestTrainingHistory = [] 
    
    X_train, _, y_train_onehot, _, _, _ = get_data(data_path)
    X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)       

    BestAcc = 0 
    for train in range(n_train):
        tf.random.set_seed(train+1)
        np.random.seed(train+1)
            
        in_run = time.time()
            
        filepath = results_path + '/saved models/run-{}'.format(train+1)
        if not os.path.exists(filepath):
            os.makedirs(filepath)        
        filepath = filepath + '/model.h5'
            
        model = getModel(model_name, dataset_conf, from_logits)
        model.compile(loss=CategoricalCrossentropy(from_logits=from_logits), optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          

        callbacks = [
            ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min'),
            ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0),
            WandbCallback()
        ]
        history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
           
        model.load_weights(filepath)
        y_pred = model.predict(X_val)
        if from_logits:
            y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
        else:
            y_pred = y_pred.argmax(axis=-1)
                
        labels = y_val_onehot.argmax(axis=-1)
        acc = accuracy_score(labels, y_pred)
                        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        out_run = time.time()


        if BestAcc < acc:
            BestAcc = acc
            bestTrainingHistory = history
        
        info = 'Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Best Acc: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc, BestAcc)
        print(info)
        log_write.write(info + '\n')

        best_models.write(filepath + '\n')
        
    if LearnCurves:
        print('Plotting Learning Curves')
        draw_learning_curves(bestTrainingHistory)
          
    out_exp = time.time()
    info = '\nAverage accuracy: {:.2f} %\nTraining time: {:.1f} min\n'.format(BestAcc * 100, (out_exp - in_exp) / 60)
    print(info)
    log_write.write(info + '\n')

    best_models.close()   
    log_write.close()

    
    
#%% Evaluation 
def test(model, dataset_conf, results_path):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    
    # Get dataset parameters
    n_classes = dataset_conf.get('n_classes')
    data_path = dataset_conf.get('data_path')
    classes_labels = dataset_conf.get('cl_labels')
     
    # Load data
    _, _, _, X_test, _, y_test_onehot = get_data(data_path)     

    # Initialize variables for accuracy and kappa scores
    acc = []
    kappa = []
    
    # Load the model
    runs = os.listdir(results_path+"/saved models")
    if len(runs) == 0:
        print("No saved models found. Please check the models directory.")
        return
    
    for run in runs:
        model_path = f'{results_path}/saved models/{run}/model.h5'
        model.load_weights(model_path)
        
        # Predict MI task
        start_time = time.time()
        y_pred = model.predict(X_test).argmax(axis=-1)
        inference_time = (time.time() - start_time) / X_test.shape[0]
        
        # Calculate accuracy and K-score          
        labels = y_test_onehot.argmax(axis=-1)
        acc.append(accuracy_score(labels, y_pred))
        kappa.append(cohen_kappa_score(labels, y_pred))
        
    # Calculate and draw confusion matrix
    cf_matrix = confusion_matrix(labels, y_pred, normalize='true')
    
    # Calculate the average performance measures 
    avg_acc = np.mean(acc) * 100
    avg_kappa = np.mean(kappa)
    info = f"\nTest performance:\n"
    info += f"-----------------\n"
    info += f"Accuracy: {avg_acc:.2f}%\n"
    info += f"Kappa Score: {avg_kappa:.3f}\n"
    info += f"Inference Time: {inference_time * 1000:.2f} ms per trial\n"
    print(info)
    log_write.write(info + '\n')
    wandb.log({"Test Accuracy": avg_acc, "Test Kappa": avg_kappa})
    # Draw confusion matrix
    draw_confusion_matrix(cf_matrix, results_path, classes_labels)
    
    log_write.close() 
    
    
#%%
def getModel(model_name, dataset_conf, from_logits = False):
    
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Select the model
    if(model_name == 'ATCNet'):
        # Train using the proposed ATCNet model: https://ieeexplore.ieee.org/document/9852687
        model = models.ATCNet_( 
            # Dataset parameters
            n_classes = n_classes, 
            in_chans = n_channels, 
            in_samples = in_samples, 
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Attention (AT) block parameter
            attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 1,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu',
            )     
    elif(model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = n_classes, Chans=n_channels, Samples=in_samples)      
    elif(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = n_classes, Chans=n_channels, Samples=in_samples)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = n_classes, Chans=n_channels, Samples=in_samples) 
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = in_samples , n_features = n_channels, n_outputs = n_classes)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)
    elif(model_name == 'MBEEG_SENet'):
        # Train using MBEEG_SENet: https://www.mdpi.com/2075-4418/12/4/995
        model = models.MBEEG_SENet(nb_classes = n_classes , Chans = n_channels, Samples = in_samples)

    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    
#%%
def run():
    # Define dataset parameters
    #dataset = 'HGD' # Options: 'BCI2a','HGD', 'CS2R'
    in_samples = 32
    n_channels = 14
    n_classes = 10
    classes_labels = [0,1,2,3,4,5,6,7,8,9]
    data_path = '/content/drive/MyDrive/EEG2Image/data/eeg/char/data.pkl'
    
        
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set dataset paramters !python main_TrainValTest.py
    dataset_conf = {'n_classes': n_classes, 'cl_labels': classes_labels,
                    'n_channels': n_channels, 'in_samples': in_samples,
                    'data_path': data_path}
    # Set training hyperparamters
    train_conf = { 'batch_size': 128, 'epochs': 100, 'patience': 100, 'n_train': 1,
                  'LearnCurves': True, 'from_logits': False, 'model':'ATCNet'}
           
    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, results_path)    

#%%
if __name__ == "__main__":
    run()
    
