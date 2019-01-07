#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antonio
@email: laguilarg@upao.edu.pe
"""

#############
# Libraries #
#############
import os
import numpy as np
import torch
import torch.nn as nn
from model_helper import dataloaders, load_model, select

####################
# Helper Functions #
####################

def evaluation(loader, classes, model):
    """
    Evaluate the training accuracy of a model.
    
    params
    ------
    loader  -- data loader.
    classes -- list containing the classes.
    model   -- trained model
    """
    model.to('cpu')
    model.eval()
    criterion = nn.NLLLoss()
    # track data loss
    data_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    # iterate over the data
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to('cpu'), target.to('cpu')
        # forward
        output = model(data)
        # loss
        loss = criterion(output, target)
        data_loss += loss.item()*data.size(0)
        # get probabilities
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        correct_tensor = top_class == target.view(*top_class.shape)
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # iterate over the labels
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average the loss
    data_loss = data_loss/len(loader.dataset)
    print('Training Loss: {:.6f}\n'.format(data_loss))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Training accuracy of the class %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Training of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTraining accuracy (overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

def main():
    model_path = '../models/'
    classes = ['back', 'done', 'left', 'right', 'up']
    model_list = list(os.listdir(model_path))
    model = load_model(select(model_list, 'Model'))
    trainloader = dataloaders()
    print('========== Training data results ===========')
    evaluation(trainloader, classes, model)

if __name__ == '__main__':
    main()
