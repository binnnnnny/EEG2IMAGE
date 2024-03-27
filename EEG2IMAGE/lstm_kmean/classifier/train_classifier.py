import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, classifier, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import wandb
import warnings

warnings.filterwarnings('ignore')

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

np.random.seed(45)
tf.random.set_seed(45)



if __name__ == '__main__':
    n_channels = 14
    n_feat = 128
    batch_size = 256
    test_batch_size = 1
    n_classes = 10

    # Load data
    with open('/content/drive/MyDrive/EEG2Image/data/eeg/char/data.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        test_X = data['x_test']
        test_Y = data['y_test']

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)     
    
    train_batch = load_complete_data(X_train, Y_train, batch_size=batch_size)
    val_batch = load_complete_data(X_valid, Y_valid, batch_size=batch_size)
    test_batch = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
    X, Y = next(iter(train_batch))
    
    # load checkpoint
    triplenet = TripleNet(n_classes=n_classes)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    triplenet_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
    triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='/content/drive/MyDrive/EEG2Image', max_to_keep=5000)
    triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
    START = int(triplenet_ckpt.step) // len(train_batch)
    if triplenet_ckptman.latest_checkpoint:
        print('Restored from the latest checkpoint, epoch: {}'.format(START))
    
    # TripleNet 모델 위에 분류 층을 추가
    classifier = classifier(triplenet,n_classes=10)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)

    START = 0 
    EPOCHS = 100
    cfreq = 178  # Checkpoint frequency
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []


    bestAcc = float('-inf')
    bestLoss = float('-inf')
    wandb.init(project='Rxde',name='EEG2IMAGE(val)')
    for epoch in range(START, EPOCHS):
        
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        train_loss = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        val_loss = tf.keras.metrics.Mean()

        tq = tqdm(train_batch)
        for idx, (X, Y) in enumerate(tq, start=1):
            Y_pred, loss = train_step(classifier, opt, X, Y)
            train_acc.update_state(Y, Y_pred)
            train_loss.update_state(loss)
            # triplenet_ckpt.step.assign_add(1)
            # if (idx % cfreq) == 0:
            #     triplenet_ckptman.save()

        tq = tqdm(val_batch)
        for idx, (X, Y) in enumerate(tq, start=1):
            val_pred, loss = test_step(classifier, X, Y)
            val_acc.update_state(Y, val_pred)
            val_loss.update_state(loss)
        train_accs.append(train_acc.result().numpy())
        train_losses.append(train_loss.result().numpy())
        val_accs.append(val_acc.result().numpy())
        val_losses.append(val_loss.result().numpy())

        print('Epoch: {}, Train Accuracy : {}, Train Loss: {}, Valdiation Accuracy : {}, Validation Loss: {}'.format(epoch, train_acc.result(),train_loss.result(), val_acc.result(),val_loss.result()))
        wandb.log({'Train Loss' : train_loss.result(),
                            "Train Accuracy" : train_acc.result(),
                            'validation loss' : val_loss.result(),
                            'validation accuracy' :  val_acc.result()})
        
        # Update
        if val_acc.result().numpy() > bestAcc :
            bestAcc = val_acc.result().numpy()
            classifier.save_weights('model.h5')

        if val_loss.result() < bestLoss :
            bestLoss = val_loss.result()
        
    print('The Average Train Accuracy : {}, The Average Train Loss: {}, The Best Valdiation Accuracy : {}, The Average Validation Accuracy : {}, The Best Validation Loss: {}, The Average Validation Loss: {}'.format(sum(train_accs) / len(train_accs), sum(train_losses) / len(train_losses), bestAcc, sum(val_accs) / len(val_accs), bestLoss, sum(val_losses) / len(val_losses)))        
    # test data
    print('\nTest performance')
    # Load the model
    model_path = 'model.h5'
    classifier.load_weights(model_path)
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    
    for X, Y in test_batch:
        Y_pred, loss = test_step(classifier, X, Y)
        test_loss.update_state(loss)
        test_acc.update_state(Y, Y_pred)
    print(f"Test Loss: {test_loss.result()}, Test Accuracy: {test_acc.result()}")
    wandb.log({'Test Loss' : test_loss.result(),
               "Test Accuracy" : test_acc.result()})
