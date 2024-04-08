"""
Transformer for EEG classification

The core idea is slicing, which means to split the signal along the time dimension. Slice is just like the patch in Vision Transformer.

"""


import os
import numpy as np
import math
import random
import time
import scipy.io

from torch.utils.data import Dataset,DataLoader,random_split
from torch.autograd import Variable
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

import pickle
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
#import wandb
from triplet_semihard_loss import *
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, predictions, labels, eval = False):
        predictions = predictions.log_softmax(dim=self.dim)
        
        with torch.no_grad():
            indicator = 1.0 - labels
            smooth_labels = torch.zeros_like(labels)
            smooth_labels.fill_(self.smoothing / (self.classes - 1))
            smooth_labels = labels * self.confidence + indicator * smooth_labels#lables->indicator

        return torch.mean(torch.sum(-smooth_labels.cuda(0) * predictions, dim=self.dim))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 3), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (14, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=1,
                 drop_p=0.5,
                 forward_expansion=2,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.reduce = Reduce('b n e -> b e', reduction='mean')
        self.norm = nn.LayerNorm(emb_size)
        self.classifier = nn.Linear(emb_size, n_classes)
       

    def forward(self, x):
        x = self.reduce(x)  
        norm_x = self.norm(x)  
        out = self.classifier(norm_x)  
        return norm_x, out
    
class ViT(nn.Sequential):
    def __init__(self, emb_size=128, depth=1, n_classes=10, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(32),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

    
class channel_attention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(14,14),
            nn.LayerNorm(14),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(14,14),
            # nn.LeakyReLU(),
            nn.LayerNorm(14),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(14,14),
            # nn.LeakyReLU(),
            nn.LayerNorm(14),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        #x = x.unsqueeze(1)
        temp = rearrange(x, 'b o c s -> b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out
    
class CustomDataset(Dataset) :
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        return img, label

class CustomDataset2(Dataset):
    def __init__(self, img, label, target):
        self.img = img
        self.label = label
        self.target = target

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        tmp_label = torch.argmax(label,dim=-1)
        target = self.target[tmp_label]  # label에 해당하는 target 가져오기
        return img, label, target
    
class Trans():
    def __init__(self):
        super(Trans, self).__init__()
        self.batch_size = 128
        self.n_epochs = 300
        #self.img_height = 22
        #self.img_width = 600
        #self.channels = 1
        self.c_dim = 4
        self.lr = 0.001
        self.lr_min = 0.00001
        self.b1 = 0.5
        self.b2 = 0.9
        self.start_epoch = 0

        self.pretrain = False

        # 로그 파일 이름을 고정 값으로 변경
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        #self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cls = LabelSmoothingLoss(10, smoothing=0.2)
        self.criterion_mse = torch.nn.MSELoss().cuda()
        self.criterion_tri = TripletLoss(device).to(device)

        self.model = ViT().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 14, 32))

        self.centers = {}


    def get_source_data(self):
    # 데이터 불러오기
        with open('/content/drive/MyDrive/EEG2Image/data/eeg/image/data.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            self.train_data = data['x_train']
            self.train_label = data['y_train']
            self.test_data = data['x_test']
            self.test_label = data['y_test']

        # standardize
        for i in range(14):
            scaler = StandardScaler()
            scaler.fit(self.train_data[:, i, :, 0] )
            self.train_data[:, i, :, 0] = scaler.transform(self.train_data[:, i, :, 0] )
            self.test_data[:, i, :, 0]  = scaler.transform(self.test_data[:, i, :, 0])

        # numpy 배열로 변환 및 차원 확장
        self.train_data = self.train_data.transpose(0, 3, 1, 2)
        self.test_data = self.test_data.transpose(0, 3, 1, 2)     
        
        return self.train_data, self.train_label, self.test_data, self.test_label

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label
    
    def phase1_train(self, model, loader, loss_CE, loss_mse, loss_tri, optimizer):
        model.train()
        loss_avg = 0.0
        for i, (input, label) in enumerate(loader): 
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # args.device : "cuda:1"
            emb,y = model(input)
            y = F.softmax(y,dim=-1)
            #y = model(input, position_embedding, mask)
            loss = loss_CE(y, label)*0.01  + loss_tri(emb,label)*0.001
            #loss = criterion(y, label)*0.1 + mse(recon, target)#*0.001
            loss.backward()
            loss_avg += loss.item()

            optimizer.step() 
            #scheduler.step()
        return loss_avg / (i + 1)
    
    def phase2_train(self, model, loader, loss_CE, loss_mse, loss_tri, optimizer, scheduler ):
        model.train()
        loss_avg = 0.0
        for i, (input, label, target) in enumerate(loader): 
            optimizer.zero_grad()
            input = input.to(device)
            label = label.to(device)
            target = target.to(device)
            # args.device : "cuda:1"
            emb,y = model(input)
            #y = F.softmax(y,dim=-1)
            #y = model(input, position_embedding, mask)
            loss = loss_CE(y, label) + loss_mse(emb,target)#*0.001
            #loss = criterion(y, label)*0.1 + mse(recon, target)#*0.001
            loss.backward()
            loss_avg += loss.item()
            #print(loss_CE(y, label))
            #print(loss_mse(emb,target))
            optimizer.step() 
            scheduler.step()
        return loss_avg / (i + 1)

    
    def phase1_evaluate(self,model, loader, loss_CE, loss_mse, loss_tri):
        model.eval()
        predictions = []
        labels = [] 
        embs = []
        loss_avg=0.0
        
        with torch.no_grad():
            for i,(input, label) in enumerate(loader):
                input = input.to(device)
                label = label.to(device)
                emb,pred = model(input)
                loss = loss_CE(pred, label) + loss_tri(emb,label)
                loss_avg += loss.item()
                prediction = F.softmax(pred, dim=-1)
                prediction = np.squeeze(prediction.detach().to("cpu").numpy())
                predictions.append(prediction)
                label = label.detach().to("cpu").numpy()
                emb = emb.detach().to("cpu").numpy()
                labels.append(label)
                embs.append(emb)

                
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        embs = np.vstack(embs)

        predictions = np.argmax(predictions, axis = -1)
        labels = np.argmax(labels, axis = -1)
        accuracy = accuracy_score(labels, predictions)

        # labels, predictions는 따로 붙임
        return accuracy, labels, predictions, embs, loss_avg / (i + 1)
    
    def phase2_evaluate(self, model, loader, loss_CE, loss_mse, loss_tri):
        model.eval()
        predictions = []
        labels = [] 
        embs = []
        loss_avg=0.0
        with torch.no_grad():
            for i,(input, label, target) in enumerate(loader):
                input = input.to(device)
                label = label.to(device)
                target = target.to(device)
                emb,pred = model(input)
                
                loss = loss_CE(pred, label) + loss_mse(emb,target)
                loss_avg += loss.item()
                prediction = F.softmax(pred, dim=-1)
                prediction = np.squeeze(prediction.detach().to("cpu").numpy())
                predictions.append(prediction)
                label = label.detach().to("cpu").numpy()
                emb = emb.detach().to("cpu").numpy()
                labels.append(label)
                embs.append(emb)

                
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        embs = np.vstack(embs)

        predictions = np.argmax(predictions, axis = -1)
        labels = np.argmax(labels, axis = -1)
        accuracy = accuracy_score(labels, predictions)

        # labels, predictions는 따로 붙임
        return accuracy, labels, predictions, embs, loss_avg / (i + 1)

    def train(self):
        # make array per class
        phase = 2
        print('The phase is ',phase)
        
        
        img, label, test_data, test_label = self.get_source_data()
        img = torch.tensor(img,dtype=torch.float32)
        label = torch.tensor(label,dtype=torch.float32)
        # 정수형으로 만들기
        #label = torch.argmax(label, dim=1)
        
        # train : validation = 8 : 2
        X_train, X_valid, Y_train, Y_valid = train_test_split(img, label, test_size=0.2, random_state=42)

        if phase == 1 :
            train_dataset = CustomDataset(X_train, Y_train)
            valid_dataset = CustomDataset(X_valid, Y_valid)
            self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
        

        if phase == 2 :
            target = np.load("target.npy")

            loop = X_train.shape[0]
            y_t = torch.argmax(Y_train, dim=1)
            X_target = []
            for i in range(loop):
                if y_t[i].item() == 0:
                    X_target.append(target[0])
                elif y_t[i].item() == 1:
                    X_target.append(target[1])
                elif y_t[i].item() == 2:
                    X_target.append(target[2])
                elif y_t[i].item() == 3:
                    X_target.append(target[3])
                elif y_t[i].item() == 4:
                    X_target.append(target[4])
                elif y_t[i].item() == 5:
                    X_target.append(target[5])
                elif y_t[i].item() == 6:
                    X_target.append(target[6])
                elif y_t[i].item() == 7:
                    X_target.append(target[7])
                elif y_t[i].item() == 8:
                    X_target.append(target[8])
                elif y_t[i].item() == 9:
                    X_target.append(target[9])


            V_target = []
            loop = X_valid.shape[0]
            y_v = torch.argmax(Y_valid, dim=1)
            for i in range(loop):
                if y_v[i].item() == 0:
                    V_target.append(target[0])
                elif y_v[i].item() == 1:
                    V_target.append(target[1])
                elif y_v[i].item() == 2:
                    V_target.append(target[2])
                elif y_v[i].item() == 3:
                    V_target.append(target[3])
                elif y_v[i].item() == 4:
                    V_target.append(target[4])
                elif y_v[i].item() == 5:
                    V_target.append(target[5])
                elif y_v[i].item() == 6:
                    V_target.append(target[6])
                elif y_v[i].item() == 7:
                    V_target.append(target[7])
                elif y_v[i].item() == 8:
                    V_target.append(target[8])
                elif y_v[i].item() == 9:
                    V_target.append(target[9])
            X_target = torch.tensor(np.array(X_target),dtype=torch.float32)
            V_target = torch.tensor(np.array(V_target),dtype=torch.float32)
            print(X_train.shape)
            print(X_target.shape)
            print(Y_train.shape)

            train_dataset = CustomDataset2(X_train, Y_train, X_target)
            valid_dataset = CustomDataset2(X_valid, Y_valid, V_target)
            self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        # test
        test_data = torch.tensor(test_data,dtype=torch.float32)
        test_label = torch.tensor(test_label,dtype=torch.float32)
        #test_label = torch.argmax(test_label, dim=1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        scheduler = CosineAnnealingLR(self.optimizer, len(self.train_dataloader) * self.n_epochs, self.lr_min)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        
        
        #wandb.init()
        bestAcc = 0
        averAcc = 0
        bestvalLoss = float('inf')
        mse_loss = 0
        num = 0

        # Train the cnn model
        total_step = len(img)
        curr_lr = self.lr
        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.
        
        
        if phase == 1 :    
            for e in range(self.n_epochs):
                in_epoch = time.time()
                loss = self.phase1_train(self.model, self.train_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri, self.optimizer)
                train_accuracy, train_label, train_pred, train_embs, train_loss = self.phase1_evaluate(self.model, self.train_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri)
                val_accuracy, val_label, val_pred, _, val_loss = self.phase1_evaluate(self.model, self.val_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri)
                
                out_epoch = time.time()

                print('Epoch:', e,
                        'Train loss:', train_loss,
                        'validation loss:' , val_loss,
                        'Train accuracy:', train_accuracy,
                        'validation accuracy is:', val_accuracy)
                

                if val_accuracy > bestAcc:
                    bestAcc = val_accuracy
                    bestvalLoss = val_loss
                    torch.save(self.model, 'phase1.pth')

        elif phase == 2:
            #wandb.init(project='Rxde',name='EEGformer_2phase_tri2')
            for e in range(self.n_epochs):
                in_epoch = time.time()
                loss = self.phase2_train(self.model, self.train_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri, self.optimizer,scheduler )
                train_accuracy, train_label, train_pred, train_embs,train_loss = self.phase2_evaluate(self.model, self.train_dataloader,self.criterion_cls, self.criterion_mse, self.criterion_tri)
                val_accuracy, val_label, val_pred,_,val_loss = self.phase2_evaluate(self.model, self.val_dataloader,self.criterion_cls, self.criterion_mse, self.criterion_tri)
                lr = scheduler.get_last_lr()[0]
                out_epoch = time.time()
      
                    
                print('Epoch:', e,
                        'Train loss:', loss,
                        'validation loss:' , val_loss,
                        'Train accuracy:', train_accuracy,
                        'validation accuracy is:', val_accuracy)

                if val_accuracy > bestAcc:
                    bestAcc = val_accuracy
                    bestvalLoss = val_loss
                    torch.save(self.model, 'phase2.pth')    
            
            """wandb.log({'Train Loss' : loss.detach().cpu().numpy(),
                        "Train Accuracy" : train_accuracy,
                        'validation loss' : val_loss.detach().cpu().numpy(),
                        'validation accuracy' :  val_accuracy})"""
            
        
                
   
        # Load Model
        if phase == 1 :
            model_path = 'phase1.pth'
        elif phase == 2 :
            model_path = 'phase2.pth'

        
        #self.model = torch.load(model_path)
        #self.model.eval()

        if phase == 1:
            self.model = torch.load("phase1.pth")
            self.model.eval()
            train_accuracy, train_label, train_pred, train_embs, train_loss = self.phase1_evaluate(self.model, self.train_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri)
            print(train_embs.shape)
            print(train_pred.shape)
            y = train_pred
            data0 = train_embs[np.squeeze(y == 0)]
            data1 = train_embs[np.squeeze(y == 1)]
            data2 = train_embs[np.squeeze(y == 2)]
            data3 = train_embs[np.squeeze(y == 3)]
            data4 = train_embs[np.squeeze(y == 4)]
            data5 = train_embs[np.squeeze(y == 5)]
            data6 = train_embs[np.squeeze(y == 6)]
            data7 = train_embs[np.squeeze(y == 7)]
            data8 = train_embs[np.squeeze(y == 8)]
            data9 = train_embs[np.squeeze(y == 9)]


            #data0 = np.where(data0 == 0, np.nan, data0)
            #data1 = np.where(data1 == 0, np.nan, data1)
            #data2 = np.where(data2 == 0, np.nan, data2)
            #data3 = np.where(data3 == 0, np.nan, data3)
            #data4 = np.where(data4 == 0, np.nan, data4)
            #data5 = np.where(data5 == 0, np.nan, data5)
            #data6 = np.where(data6 == 0, np.nan, data6)
            #data7 = np.where(data7 == 0, np.nan, data7)
            #data8 = np.where(data8 == 0, np.nan, data8)
            #data9 = np.where(data9 == 0, np.nan, data9)
            
            
            target=np.zeros([10,128])
            target[0,:] = np.mean(data0, axis=0)
            target[1,:] = np.mean(data1, axis=0)
            target[2,:] = np.mean(data2, axis=0)
            target[3,:] = np.mean(data3, axis=0)
            target[4,:] = np.mean(data4, axis=0)
            target[5,:] = np.mean(data5, axis=0)
            target[6,:] = np.mean(data6, axis=0)
            target[7,:] = np.mean(data7, axis=0)
            target[8,:] = np.mean(data8, axis=0)
            target[9,:] = np.mean(data9, axis=0)
            plt.figure(figsize=(15, 15))
            plt.plot(target[0])
            plt.plot(target[1])
            plt.plot(target[2])
            plt.plot(target[3])
            plt.plot(target[4])
            plt.plot(target[5])
            plt.plot(target[6])
            plt.plot(target[7])
            plt.plot(target[8])
            plt.plot(target[9])
            # plt.legend(loc='lower right')
            plt.savefig("class1.jpg")
            plt.clf()
            #print(target)

            np.save("target.npy",target)

        
        print('################ inference ################')

        with torch.no_grad():
            self.model = torch.load("phase2.pth")
            self.model.eval()
            test_accuracy, test_label, test_pred, test_emb, test_loss = self.phase1_evaluate(self.model, self.test_dataloader, self.criterion_cls, self.criterion_mse, self.criterion_tri)
            tsne = TSNE(n_components=2)
            colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
            vis = tsne.fit_transform(test_emb)
            y = test_label
            data0 = vis[np.squeeze(y == 0)]
            data1 = vis[np.squeeze(y == 1)]
            data2 = vis[np.squeeze(y == 2)]
            data3 = vis[np.squeeze(y == 3)]
            data4 = vis[np.squeeze(y == 4)]
            data5 = vis[np.squeeze(y == 5)]
            data6 = vis[np.squeeze(y == 6)]
            data7 = vis[np.squeeze(y == 7)]
            data8 = vis[np.squeeze(y == 8)]
            data9 = vis[np.squeeze(y == 9)]
            plt.figure(figsize=(10,10))
            plt.scatter(data0[:, 0], data0[:, 1], c=colors[0])
            plt.scatter(data1[:, 0], data1[:, 1], c=colors[1])
            plt.scatter(data2[:, 0], data2[:, 1], c=colors[2])
            plt.scatter(data3[:, 0], data3[:, 1], c=colors[3])
            plt.scatter(data4[:, 0], data4[:, 1], c=colors[4])
            plt.scatter(data5[:, 0], data5[:, 1], c=colors[5])
            plt.scatter(data6[:, 0], data6[:, 1], c=colors[6])
            plt.scatter(data7[:, 0], data7[:, 1], c=colors[7])
            plt.scatter(data8[:, 0], data8[:, 1], c=colors[8])
            plt.scatter(data9[:, 0], data9[:, 1], c=colors[9])
            plt.savefig("tsne.jpg")
            plt.clf()

        print('The test accuracy is:', test_accuracy)
        print("The test loss :", test_loss)
        
        if phase == 2 :
            """wandb.log({'Test Loss': test_loss, 
                        'Test Accuracy': test_accuracy})
        
            wandb.finish()"""
        return bestAcc, bestvalLoss, test_accuracy, test_loss
    

    
    
def main():
    #seed_n = np.random.randint(500)
    seed_n = 78
    print('seed is ' + str(seed_n))
   
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    trans = Trans()
    bestAcc, bestvLoss, bestTAcc, bestTLoss = trans.train()
    print('THE Validation ACCURACY IS ' + str(bestAcc))
    print('THE Validation LOSS IS ' + str(bestvLoss))
    print('THE Test ACCURACY IS ' + str(bestTAcc))
    print('THE Test LOSS IS ' + str(bestTLoss))

    

if __name__ == "__main__":
    main()

