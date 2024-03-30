import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
import skimage.io as skio
from skimage.transform import resize
from tifffile import imwrite
import numpy as np
import cv2
from save_figure import save_figure, save_figure_condition
import h5py
from functools import partial
import tensorflow_io as tfio
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import style
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

style.use('seaborn')

class CustomDataset(Dataset):
    def __init__(self, X, Y, paths, resolution=128, transform=None):
        self.X = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)  # CHW 형식으로 변환
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.Y = torch.argmax(self.Y, dim=1)
        self.paths = paths
        self.resolution = resolution
        
        self.default_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform = transform if transform is not None else self.default_transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.Y[idx]
        path = self.paths[idx]

        # 이미지 로딩 및 전처리
        image = Image.open(path).convert('RGB')  
        image = self.transform(image)

        return data, label, image
	
def load_complete_data(X, Y, P, batch_size=16, dataset_type='train', transform=None):
    # 학습용 데이터셋 및 DataLoader 생성
    if dataset_type == 'train':
        
        if not transform:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        dataset = CustomDataset(X, Y, P, transform=transform)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
   
    else:
        
        if not transform:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        dataset = CustomDataset(X, Y, P, transform=transform)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def show_batch_images(X, save_path, Y=None):
	# Y = np.squeeze(tf.argmax(Y, axis=-1).numpy())
	X = np.clip( np.uint8( ((X.numpy() * 0.5) + 0.5) * 255 ), 0, 255)
	# X = X[:16]
	col = 4
	row = X.shape[0] // col
	# print(X.shape[0], Y.shape)
	for r in range(row):
          for c in range(col):
               plt.subplot2grid((row, col), (r, c), rowspan=1, colspan=1)
               plt.grid('off')
               plt.axis('off')
               if Y is not None:
                    plt.title('{}'.format(Y[r*col+c]))
               plt.imshow(X[r*col+c])
               
	plt.tight_layout()
	plt.savefig(save_path)
	plt.clf()
	plt.close()