#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:48 2020

@author: naveenpaluru
"""
import sklearn.metrics as metrics
import os,time
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import torch.nn as nn
from datetime import datetime
from config import Config
from mydataset import myDataset
#import h5py as h5
from enet import ENet
import torch.optim as optim
from scipy.io import loadmat
import torch.nn.functional as F
from torch.optim import lr_scheduler

print ('********************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model'
print('Model will be saved to  : ', directory)

if not os.path.exists(directory):
    os.makedirs(directory)

config  = Config()

if config.gpu==True:
    torch.cuda.manual_seed(9001)
else:
    torch.manual_seed(9001)
    

# load the  data
data = loadmat('dataset250.mat')

# get training data
trainimgs = data['trainimgs']
trainlabl = data['trainlabel']

# get validation data
valimgs   = data['valimgs']
vallabels = data['vallabel']



transform = transforms.Compose([transforms.ToTensor()])

# make the data iterator for training data
train_data = myDataset(trainimgs, trainlabl, transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.trainbatchsize, shuffle=True, num_workers=2)


# make the data iterator for validation data
valid_data = myDataset(valimgs, vallabels, transform)
valloader  = torch.utils.data.DataLoader(valid_data, batch_size=config.valbatchsize, shuffle=False, num_workers=2)

print('----------------------------------------------------------')
#%%
# Create the object for the network

if config.gpu == True:
    net = ENet()
    net.cuda(config.gpuid)
else:
     net = ENet()
    

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Define the loss function
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

# Iterate over the training dataset
train_loss = []
val_loss   = []

for j in range(config.epochs):  
    running_trainloss = 0
    running_valloss   = 0
    # Start epochs   
   
    net.train() 
    for i,data in tqdm.tqdm(enumerate(trainloader)): 
        # start iterations
        images,labels = Variable(data[0]),Variable(data[1])
        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
        # make forward pass      
        output = net(images)
        #output = torch.unsqueeze(torch.squeeze(output),dim=1)
        #compute loss
        loss   = criterion(output, labels.squeeze())
        # make gradients zero
        optimizer.zero_grad()
        # back propagate
        loss.backward()
        running_trainloss += loss.item()
        # update the parameters
        optimizer.step()
    # print loss after every epoch
    
    print('\nTraining - Epoch {}/{}, loss:{:.4f}'.format(j+1, config.epochs, running_trainloss/len(trainloader)))
    train_loss.append(running_trainloss/len(trainloader))
    
    net.eval()    
    
    for i,data in tqdm.tqdm(enumerate(valloader)): 
        
        # start iterations
        images,labels = Variable(data[0]),Variable(data[1])
        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
        # make forward pass      
        output = net(images)
        
        #compute loss
        loss   = criterion(output, labels.squeeze())
        running_valloss += loss.item()
        
        output = F.softmax(output,dim=1)
        output = output.cpu().detach()
        labels = labels.cpu().detach()
        
        if i==0:
            pred =output
            truth=labels
        else:
            pred   = torch.cat((pred, output),0)
            truth  = torch.cat((truth,labels),0)
            
    matrix = metrics.confusion_matrix(pred.argmax(axis=1).numpy(), truth.numpy())
    acc = np.sum(np.diag(matrix))/(len(valloader)*config.valbatchsize)
      
    # print validation loss after every epoch
    
    print('\nValidatn - Epoch {}/{}, loss:{:.4f}, Acc:{:.4f}'. format(j+1, config.epochs, running_valloss/len(valloader), acc))
    print('-----------------------------------------------------------------------------------')
    val_loss.append(running_valloss/len(valloader))
  

    #save the model      
    torch.save(net.state_dict(),os.path.join(directory,"ENet" + str(j+1) +"_model.pth"))
    
    scheduler.step()
# plot the training and validation loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.xlabel('epochs')
plt.ylabel('Train Loss ')   
plt.show()
plt.figure()
plt.plot(x, val_loss ,label='Validatn')
plt.xlabel('epochs')
plt.ylabel('Val Loss ')                                  
plt.show()

