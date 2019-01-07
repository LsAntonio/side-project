#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antonio
@email: laguilarg@upao.edu.pe
"""
#############
# Libraries #
#############

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import models, datasets, transforms
from collections import OrderedDict

np.random.seed(5)

####################
# Helper Functions #
####################

def dataloaders():
    """
    Define transforms for the training data.
    
    returns
    -------
    trainloader -- training data loader.
    """
    # train data path
    data_train = '../dataset/train/'
    # set transformations
    train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(data_train, transform = train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)
    
    return trainloader


def get_model(training = True):
    """
    Define a pretrained model to use.
    
    params
    ------
    training  -- True for training mode, False for test mode.
    
    returns
    -------
    model -- a model for training or test.
    """
    # download the model
    model = models.alexnet(pretrained = True)
    # define the FC layers
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 2024)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p = .7)),
        ('fc2', nn.Linear(2024, 516)),
        ('relu', nn.ReLU()),
        ('dropput', nn.Dropout(p = .7)),
        ('fc3', nn.Linear(516, 5)),
        ('output', nn.LogSoftmax(dim=1))]))
    # configure the last layer
    model.classifier = classifier
    if training:
        # set model training parameters
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = 1e-4, amsgrad = True)
        return  model, criterion, optimizer
    else:
        return model

def select(files, file_type):
    """
    Select a file name.
    
    params
    ------
    files     -- file directory list
    file_type -- file type to display.
    
    returns
    -------
    file      -- a file name.
    """
    k = 0
    print('== ' + file_type + ' List ==')
    for file in files:
        print("[" + str(k) + "]. " + file)
        k += 1
    print('Select a ' + file_type + ' to continue')
    idx = int(input())
    return files[idx]


def load_model(model_name):
    """
    Load a trained model
    
    params
    ------
    model_name -- name of the model.
    
    returns
    -------
    model      -- trained model.
    """
    model = get_model(training = False)
    checkpoint = torch.load('../models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def preprocess(img):
    """
    Preprocess an image to feed the model.
    params
    -----
    img -- PIL image.
    
    returns
    -------
    img -- preprocessed tensor image.
    """
    # standard mean and std for the model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # resize
    img = img.resize(size = (224, 224))
    # transforms to numpy
    img = np.array(img, dtype = np.float64)
    # Mean and Std
    img = (img - mean)/std
    # transpose [channels first]
    img = img.transpose((2, 0, 1))
    # conver to Tensor
    img = torch.from_numpy(img)
    return img
