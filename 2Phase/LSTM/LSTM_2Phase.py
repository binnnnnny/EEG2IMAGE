import wandb
import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, classifier, train_step, test_step, target_step, test_step2
from utils import load_complete_data,load_complete_data2
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


# Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

if __name__ == '__main__':
    n_channels = 14
    n_feat = 128
    batch_size = 256
    test_batch_size = 1
    n_classes = 10
    phase=2


    # make array per class
    label_0=[]
    label_1=[]
    label_2=[]
    label_3=[]
    label_4=[]
    label_5=[]
    label_6=[]
    label_7=[]
    label_8=[]
    label_9=[]

    # Load data
    with open('/content/drive/MyDrive/EEG2IMAGE/EEG2Image-main/data/eeg/image/data.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        test_X = data['x_test']
        test_Y = data['y_test']

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)


    if phase == 2:
      target = np.load("target.npy")
      loop = X_train.shape[0]
      X_target = []
      y = np.argmax(Y_train,axis=-1)
      for i in range(loop):
        if y[i] == 0:
          X_target.append(target[0])
        elif y[i] == 1:
          X_target.append(target[1])
        elif y[i] == 2:
          X_target.append(target[2])
        elif y[i] == 3:
          X_target.append(target[3])
        elif y[i] == 4:
          X_target.append(target[4])
        elif y[i] == 5:
          X_target.append(target[5])
        elif y[i] == 6:
          X_target.append(target[6])
        elif y[i] == 7:
          X_target.append(target[7])
        elif y[i] == 8:
          X_target.append(target[8])
        elif y[i] == 9:
          X_target.append(target[9])

      loop = X_valid.shape[0]
      V_target = []
      y = np.argmax(Y_valid,axis=-1)
      for i in range(loop):
        if y[i] == 0:
          V_target.append(target[0])
        elif y[i] == 1:
          V_target.append(target[1])
        elif y[i] == 2:
          V_target.append(target[2])
        elif y[i] == 3:
          V_target.append(target[3])
        elif y[i] == 4:
          V_target.append(target[4])
        elif y[i] == 5:
          V_target.append(target[5])
        elif y[i] == 6:
          V_target.append(target[6])
        elif y[i] == 7:
          V_target.append(target[7])
        elif y[i] == 8:
          V_target.append(target[8])
        elif y[i] == 9:
          V_target.append(target[9])

    train_batch = load_complete_data(X_train, Y_train, batch_size=batch_size)
    val_batch = load_complete_data(X_valid, Y_valid, batch_size=batch_size)
    test_batch = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
    X, Y = next(iter(train_batch))

    if phase == 2:
      train_batch = load_complete_data2(X_train, Y_train, X_target, batch_size=batch_size)
      val_batch = load_complete_data2(X_valid, Y_valid, V_target, batch_size=batch_size)

      X, Y, T = next(iter(train_batch))



    # # load checkpoint
    triplenet = TripleNet(n_classes=n_classes)


    # TripleNet 모델 위에 분류 층을 추가
    classifier = classifier(triplenet,n_classes=10)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)

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
        if phase == 1:
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
              classifier.save_weights('model1.h5')

          if val_loss.result() < bestLoss :
              bestLoss = val_loss.result()

        elif phase == 2:
          tq = tqdm(train_batch)
          for idx, (X, Y, T) in enumerate(tq, start=1):
              Y_pred, loss = target_step(classifier, opt, X, Y, T)
              train_acc.update_state(Y, Y_pred)
              train_loss.update_state(loss)
              # triplenet_ckpt.step.assign_add(1)
              # if (idx % cfreq) == 0:
              #     triplenet_ckptman.save()

          tq = tqdm(val_batch)
          for idx, (X, Y, T) in enumerate(tq, start=1):
              val_pred,loss = test_step2(classifier, X, Y, T)
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
              classifier.save_weights('model2.h5')

          if val_loss.result() < bestLoss :
              bestLoss = val_loss.result()

    print('The Average Train Accuracy : {}, The Average Train Loss: {}, The Best Valdiation Accuracy : {}, The Average Validation Accuracy : {}, The Best Validation Loss: {}, The Average Validation Loss: {}'.format(sum(train_accs) / len(train_accs), sum(train_losses) / len(train_losses), bestAcc, sum(val_accs) / len(val_accs), bestLoss, sum(val_losses) / len(val_losses)))
    # test data
    print('\nTest performance')

    # Load the model
    if phase == 1:
       model_path = 'model1.h5'
    elif phase == 2:
       model_path = 'model2.h5'

    classifier.load_weights(model_path)
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    train_batch = load_complete_data(X_train, Y_train, batch_size=1)

    for X, Y in test_batch:
        Y_pred, loss = test_step(classifier, X, Y)
        test_loss.update_state(loss)
        test_acc.update_state(Y, Y_pred)

    if phase == 1:
      for X, Y in train_batch:
          Y_emb,Y_pred = test_step(classifier, X, Y)
          pred_label =tf.argmax(Y_pred, axis= 1)
          if pred_label == 0:
              label_0.append(Y_emb)
          elif pred_label == 1:
              label_1.append(Y_emb)
          elif pred_label == 2:
              label_2.append(Y_emb)
          elif pred_label == 3:
              label_3.append(Y_emb)
          elif pred_label == 4:
              label_4.append(Y_emb)
          elif pred_label == 5:
              label_5.append(Y_emb)
          elif pred_label == 6:
              label_6.append(Y_emb)
          elif pred_label == 7:
              label_7.append(Y_emb)
          elif pred_label == 8:
              label_8.append(Y_emb)
          elif pred_label == 9:
              label_9.append(Y_emb)

      target=np.zeros([10,128])
      print(np.array(label_0).shape)

      target[0,:] = np.mean(np.array(label_0),axis=0)
      target[1,:] = np.mean(np.array(label_1),axis=0)
      target[2,:] = np.mean(np.array(label_2),axis=0)
      target[3,:] = np.mean(np.array(label_3),axis=0)
      target[4,:] = np.mean(np.array(label_4),axis=0)
      target[5,:] = np.mean(np.array(label_5),axis=0)
      target[6,:] = np.mean(np.array(label_6),axis=0)
      target[7,:] = np.mean(np.array(label_7),axis=0)
      target[8,:] = np.mean(np.array(label_8),axis=0)
      target[9,:] = np.mean(np.array(label_9),axis=0)


      np.save("target.npy",target)

    print(f"Test Loss: {test_loss.result()}, Test Accuracy: {test_acc.result()}")

    wandb.log({'Test Loss' : test_loss.result(),
               "Test Accuracy" : test_acc.result()})

    # TSNE 시각화
    kmeanacc = 0.0
    tq = tqdm(test_batch)
    feat_X = []
    feat_Y = []
    for idx, (X, Y) in enumerate(tq, start=1):
      _, feat = classifier(X, training=False)
      feat_X.extend(feat.numpy())
      feat_Y.extend(Y.numpy())
    feat_X = np.array(feat_X)
    feat_Y = np.array(feat_Y)
    print(feat_X.shape, feat_Y.shape)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
    tsne_results = tsne.fit_transform(feat_X)
    df = pd.DataFrame()
    df['label'] = feat_Y
    df['x1'] = tsne_results[:, 0]
    df['x2'] = tsne_results[:, 1]
    # df['x3'] = tsne_results[:, 2]
    df.to_csv('experiments/inference/triplet_embed2D.csv')


    df = pd.read_csv('experiments/inference/triplet_embed2D.csv')

    plt.figure(figsize=(16,10))

    sns.scatterplot(
        x="x1", y="x2",
        data=df,
        hue='label',
        palette=sns.color_palette("hls", n_classes),
        legend="full",
        alpha=0.4
        )

    plt.show()

    kmeans = KMeans(n_clusters=n_classes,random_state=45)
    kmeans.fit(feat_X)
    labels = kmeans.labels_
    # print(feat_Y, labels)
    correct_labels = sum(feat_Y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, feat_Y.shape[0]))
    kmeanacc = correct_labels/float(feat_Y.shape[0])
    print('Accuracy score: {0:0.2f}'. format(kmeanacc))
