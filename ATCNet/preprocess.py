import numpy as np
import scipy.io as sio
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


#%%
def load_data(data_path): 
    with open(data_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']

    return X_train, y_train, X_test, y_test

#%%
def get_data(path, classes_labels = 'all', isStandard = True, isShuffle = True):
    
    X_train, y_train, X_test, y_test = load_data(path)
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)
  
    # Prepare training data  
    X_train =  X_train.transpose(0,3,1,2)
    y_train_onehot = y_train
    y_train = np.argmax(y_train, axis=1)

    # Prepare testing data 
    X_test =  X_test.transpose(0,3,1,2)
    y_test_onehot = y_test
    y_test = np.argmax(y_test,axis=1)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

