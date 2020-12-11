#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:00:26 2020

@author: naveenpaluru
"""


import numpy as np
import torch

class Config :    
    def __init__(self):     
        self.gpu       =True
        self.gpuid     = 1
        self.trainbatchsize = 5
        self.testbatchsize  = 1
        self.valbatchsize   = 5
        self.epochs    = 20
        
        
        