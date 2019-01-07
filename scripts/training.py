#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antonio
@email: laguilarg@upao.edu.pe
"""
#############
# Libraries #
#############

import torch
import numpy as np
import pandas as pd
import os
from model_helper import get_model, dataloaders, select

np.random.seed(5)
torch.manual_seed(7)


####################
# Helper Functions #
####################

def train_model(epochs = 20):
    """
    Train a model.
    
    params
    ------
    epochs -- number of epochs to train.
    
    return
    ------
    model  -- Trained model.
    """
    # get the model with its parameters
    model, criterion, optimizer = get_model(training = True)
    # track training loss
    g_training_loss = []
    g_validation_loss = []
    g_epochs = []
    # set initial validation loss
    valid_loss_min = np.inf
    current_epoch = 0
    # load the data
    trainloader = dataloaders()
    valloader = trainloader
    model.to('cpu');

    for epoch in range(1, epochs + 1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # set the model to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to('cpu'), target.to('cpu')
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
           
        # validate the model
        model.eval()
        for batch_idx, (data, target) in enumerate(valloader):
            data, target = data.to('cpu'), target.to('cpu')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
                
        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = valid_loss/len(valloader.dataset)
        # saving losses
        g_training_loss.append(train_loss)
        g_validation_loss.append(valid_loss)
        g_epochs.append(current_epoch + epoch)

        print('Epoch: {} \tEpoch Loss: {:.6f} \tTotal Training Loss: {:.6f}'.format(
                current_epoch + epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
            # save model if validation decreases
            torch.save({
                    'epoch': current_epoch + epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss}, '../models/supertuxkart_best_epoch_' + 
            str(current_epoch + epoch) + '_loss_' + str(valid_loss) + '.pth.tar')
            # Update the current loss
            valid_loss_min = valid_loss
    # store loss history
    temp = pd.DataFrame()
    temp['epochs'] = g_epochs
    temp['epoch_loss'] = g_training_loss
    temp['training_loss'] = g_validation_loss
    # save loss
    temp.to_csv('../plots/model_log_' + str(current_epoch + epoch) + '.csv', sep = ',', index = False)


def main():
    print('######### Configuration #########')
    # set the number of epochs
    epochs = int(input('Insert the number of epochs: '))
    print('Beginning Training...')
    # train the model.
    train_model(epochs = epochs)
    print('Done!')

if __name__ == '__main__':
    main()