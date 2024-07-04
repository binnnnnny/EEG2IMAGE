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

from torch.utils.data import DataLoader,random_split
from torch.autograd import Variable
from torchsummary import summary

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

import pickle
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#from common_spatial_pattern import csp
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

import matplotlib.pyplot as plt
import wandb
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

#writer = SummaryWriter('./TensorBoardX/')

# torch.cuda.set_device(6)
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


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
            Rearrange('b e (h) (w) -> b (h w) e'), # compressing(1xT)
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



class Trans():
    def __init__(self):
        super(Trans, self).__init__()
        self.batch_size = 128
        self.n_epochs = 100
        
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.start_epoch = 0

        self.pretrain = False

        # 로그 파일 이름을 고정 값으로 변경
        self.log_write = open("/content/drive/MyDrive/log.txt", "w")

        #self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 14, 32))

        self.centers = {}


    def get_source_data(self):
    # Load Data
        with open('/content/drive/MyDrive/EEG2Image/data/eeg/char/data.pkl', 'rb') as file:
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

    def train(self):
        wandb.init(project='Rxde',name='EEGformer_std2')
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        
        # 정수형으로 만들기
        label = torch.argmax(label, dim=1)
        
        # train : validation = 8 : 2
        dataset_size = len(img)
        print(dataset_size)
        train_size = int(dataset_size * 0.8)
        print(train_size)
        valid_size = dataset_size - train_size 
        print(valid_size)

        dataset = torch.utils.data.TensorDataset(img, label)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Dataloader
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)


        # test
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_label = torch.argmax(test_label, dim=1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        wandb.init()
        bestAcc = 0
        averAcc = 0
        bestvalLoss = float('inf')
        num = 0
        
        total_step = len(dataset)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for train_data, train_label in self.train_dataloader:

                train_data = Variable(train_data.cuda().type(self.Tensor))
                train_label = Variable(train_label.cuda().type(self.LongTensor))
                tok, outputs = self.model(train_data)
                loss = self.criterion_cls(outputs, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                self.model.eval()
                for valid_data, valid_label in self.val_dataloader:

                    valid_data = Variable(valid_data.cuda().type(self.Tensor))
                    valid_label = Variable(valid_label.cuda().type(self.LongTensor))
                
                    Tok, Cls = self.model(valid_data)

                    loss_valid = self.criterion_cls(Cls, valid_label)
                    y_pred = torch.max(Cls, 1)[1]
                    valid_acc = float((y_pred == valid_label).cpu().numpy().astype(int).sum()) / float(valid_label.size(0))
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == train_label).cpu().numpy().astype(int).sum()) / float(train_label.size(0))
                
                print('Epoch:', e,
                        'Train loss:', loss.detach().cpu().numpy(),
                        'validation loss:', loss_valid.detach().cpu().numpy(),
                        'Train accuracy:', train_acc,
                        'validation accuracy is:', valid_acc)
                
                wandb.log({'Train Loss' : loss.detach().cpu().numpy(),
                            "Train Accuracy" : train_acc,
                            'validation loss' : loss_valid.detach().cpu().numpy(),
                            'validation accuracy' :  valid_acc})
                    
                self.log_write.write(str(e) + "    " + str(valid_acc) + "\n")
                num = num + 1
                averAcc = averAcc + valid_acc
                
                if valid_acc > bestAcc:
                    bestAcc = valid_acc
                    
                    torch.save(self.model, 'model_complete.pth')

                if loss_valid.item() < bestvalLoss :
                    bestvalLoss = loss_valid.item()
                    
                
                torch.save(self.model, 'model_complete.pth')
        
        torch.save(self.model, 'model_complete.pth')
        averAcc = averAcc / num
        print('The average validation accuracy is:', averAcc)
        print('The best validation accuracy is:', bestAcc)
        print("The best validation loss :", bestvalLoss)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        
        print('################ inference ################')
        self.model.eval()  # 모델을 평가 모드로 설정
        bestTAcc = 0
        averTAcc = 0
        #averloss = 0
        bestTestLoss = float('inf')
        num = 0

        with torch.no_grad():
            Tok, Cls = self.model(test_data)
            loss_test = self.criterion_cls(Cls, test_label)
            y_pred = torch.max(Cls, 1)[1]
            test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
    
        averTAcc = averTAcc + test_acc

        if averTAcc > bestTAcc :
            bestTAcc = test_acc
        if loss_test.item() < bestTestLoss :
            bestTestLoss = loss_test.item()
        
        if loss_test.item() < bestTestLoss :
            bestTestLoss = loss_test.item()
        
        print('The average test accuracy is:', averTAcc)
        print('The best test accuracy is:', bestTAcc)
        print("The best test loss :", bestTestLoss)

        wandb.log({'Test Loss': loss_test, 
                   'Test Accuracy': test_acc})
        
        wandb.finish()
        return bestAcc, averAcc, bestTAcc, averTAcc
    

    
    
def main():
    # wandb config
    best = 0
    aver = 0

    seed_n = np.random.randint(500)
    print('seed is ' + str(seed_n))
    # wandb config
    config  = {
        'epochs': 100,
        'classes':10,
        'batch_size': 128,
        'learning_rate': 0.0002,
        'depth' : 1,
        'architecture': 'EEGformer',
        'seed': seed_n
        }
    
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    trans = Trans()
    bestAcc, averAcc, bestTAcc, averTAcc = trans.train()
    print('THE BEST Validation ACCURACY IS ' + str(bestAcc))
    print('THE Validation average ACCURACY IS ' + str(averAcc))
    print('THE BEST Test ACCURACY IS ' + str(bestTAcc))
    print('THE Test average ACCURACY IS ' + str(averTAcc))

    with open('/content/drive/MyDrive/sub_result.txt','w') as result_write :
        result_write.write('Seed is: ' + str(seed_n) + "\n")
        result_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('The average accuracy is: ' + str(averAcc) + "\n")    
        result_write.close()
    

if __name__ == "__main__":
    main()

