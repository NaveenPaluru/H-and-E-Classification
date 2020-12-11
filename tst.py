#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:04:43 2020

@author: naveenpaluru
"""

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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot 
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import torch.nn.functional as F



def test(directory):
    net = ENet()
    net.load_state_dict(torch.load(directory))
    
    config  = Config()
    # load the  data
    data = loadmat('dataset250.mat')
    
    # get testing data
    testimgs = data['testimgs']
    testlabl = data['testlabel']
    
    
   
    
    transform = transforms.Compose([transforms.ToTensor(),
               ])
    
        
    # make the data iterator for testing data
    test_data = myDataset(testimgs, testlabl, transform)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=config.testbatchsize, shuffle=False, num_workers=2)
    
    
     
    if config.gpu == True:
        net.cuda(config.gpuid)
        net.eval()
    else:
        net.eval()
    
    imageprob = []
    patienpro = []
    imagelabs = []
    patientlb = []
    start1 = 0
    """ 
    Note that the mini batch size of test data is 1 x 3 x 250 x 250, like this 11 RGB images belong to
    to one patient i.e. for every batch we need to accumulate image level probabilities and for every 
    11th batch we need to accumulate patient level probabilities. The following code sample is being
    written using the same logic.Note that taking the mean of similar values does not change the value.
    
     
    """
    
    for i,data in tqdm.tqdm(enumerate(testloader)): 
            
        # start iterations
        images,labels = Variable(data[0]),Variable(data[1])
        # ckeck if gpu is available
        if config.gpu == True:
            images = Variable(images.cuda(config.gpuid))
            labels = Variable(labels.cuda(config.gpuid))
        # make forward pass     
        out    = net(images)
        output = F.softmax(out ,dim=1)      
                 
        output = output.cpu().detach()
        labels = labels.cpu().detach().float()
        
        if i==0:
            pred =output
            truth=labels
            
        else:
            pred   = torch.cat((pred, output),0)
            truth  = torch.cat((truth,labels),0)
           
            
        if np.remainder(i+1, 11) == 0:
            
            if i+1 == 11:
                predp = torch.mean(pred[start1:start1+ 11,:],dim=0).view(-1,2)
                truthp= torch.mean(truth[start1:start1+11,:],dim=0).view(-1,1)
                start1=start1+11
            else:
                tempp = torch.mean(pred[start1:start1 + 11,:],dim=0).view(-1,2)
                templ = torch.mean(truth[start1:start1 +11,:],dim=0).view(-1,1)
                start1=start1+11
                predp   = torch.cat((predp, tempp),0)
                truthp  = torch.cat((truthp,templ),0)    
            
    
     
    # Plot the ROC Curves at image level
    
    print('\n')
    print('Image Level Statistics\n')
            
    # generate a no skill prediction (majority class)
    imagelabs = truth.numpy() ##
    imageprob = pred[:,1].numpy()
    ns_probs = [0 for _ in range(len(imagelabs))]
    
    # calculate scores
    ns_auc = roc_auc_score(imagelabs,  ns_probs)
    lr_auc = roc_auc_score(imagelabs, imageprob)
    
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('CNN Pred: ROC AUC=%.3f' % (lr_auc))

    
    #calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(imagelabs, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(imagelabs, imageprob)
    plt.figure()
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, linestyle='--', label='CNN')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    
    matrix = metrics.confusion_matrix(pred.argmax(axis=1).numpy(), truth.numpy())
    acc = np.sum(np.diag(matrix))/(len(testloader)*config.testbatchsize)
   
    print('Acc : = %.3f' %(acc))         
       
    plt.figure()
    df_cm = pd.DataFrame(matrix, index = ['Normal', 'Failure'],
                               columns = ['Normal', 'Failure'])
    #plt.figure(figsize = (10,10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt = 'd') # font sizesn.set(font_scale=1.4) # for label size

    
    plt.tight_layout()
    plt.xlabel('Target Class')    
    plt.ylabel('Output Class') 
    plt.show()
    
    print('---------------------------------')
    # Plot the ROC Curves at Patient level
    
    print('Patient Level Statistics\n')
            
    # generate a no skill prediction (majority class)
    
    patientlb = truthp.numpy() ##
    patienpro = predp[:,1].numpy()
    ns_probs = [0 for _ in range(len(patientlb))]
    
    # calculate scores
    ns_auc = roc_auc_score(patientlb,  ns_probs)
    lr_auc = roc_auc_score(patientlb, patienpro)
    
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('CNN Pred: ROC AUC=%.3f' % (lr_auc))
       
    #calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(patientlb,  ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(patientlb, patienpro)
    plt.figure()
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, linestyle='--', label='CNN')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    
    matrix = metrics.confusion_matrix(predp.argmax(axis=1).numpy(), truthp.numpy())
    acc = np.sum(np.diag(matrix))/(len(testloader)/11)
   
    print('Acc : = %.3f' %(acc))         
       
    plt.figure()
    df_cm = pd.DataFrame(matrix, index = ['Normal', 'Failure'],
                               columns = ['Normal', 'Failure'])
    #plt.figure(figsize = (10,10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt = 'd') # font sizesn.set(font_scale=1.4) # for label size

    
    plt.tight_layout()
    plt.xlabel('Target Class')    
    plt.ylabel('Output Class') 
    plt.show()
    
    

saveDir='./savedModels/'
cwd=os.getcwd()

# if want to test on a specific model
directory=saveDir+"ENet20_model.pth"
print('Loading the Model : ', directory)
test(directory)








































