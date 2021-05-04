# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:39:33 2021

@author: Jacob
"""


import torch 
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim
from torchvision import transforms
import time
 
# 定义CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3), # 16, 26 ,26
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),# 32, 24, 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2)) # 32, 12,12     (24-2) /2 +1
        
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3), # 64,10,10
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3),  # 128,8,8
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))  # 128, 4,4
        
        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,10))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
 
 
# 使用内置函数下载mnist数据集
train_set = mnist.MNIST('./data',train=True,download=True)
test_set = mnist.MNIST('./data',train=False,download=True)
 
# 预处理=>将各种预处理组合在一起
data_tf = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5],[0.5])])
 
train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=True)
 
train_data = DataLoader(train_set,batch_size=64,shuffle=True)
test_data = DataLoader(test_set,batch_size=128,shuffle=False)
 
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),1e-4)
 
nums_epoch = 20
 
# 开始训练
losses =[]
acces = []
eval_losses = []
eval_acces = []
res = []

for epoch in range(nums_epoch):
    time_start = time.time()
    train_loss = 0
    train_acc = 0
    net = net.train()
    for img , label in train_data:
        #img = img.reshape(img.size(0),-1) 
        img = Variable(img)
        label = Variable(label)
        
        # 前向传播
        out = net(img)
        loss = criterion(out,label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
       
        train_acc += acc
        
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img , label in test_data:
        #img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)
        
        out = net(img)
        
        loss = criterion(out,label)
        
        # 记录误差
        eval_loss += loss.item()
        
        _ , pred = out.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    
    print('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
        epoch+1, train_loss / len(train_data),train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data)))
    time_end = time.time()
    res.append([epoch+1, train_loss / len(train_data),train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data), time_end - time_start])

import csv
f = open('MNIST CNN2 Result timing lr0.0001 .csv','a',newline='')

res_writer = csv.writer(f)

for i in res:
    res_writer.writerow(i)

f.close()
 