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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
sea.set()

####################
# Helper Functions #
####################

def select(files):
    """
    Select a history loss to display.
    
    params
    ------
    files     -- list of files in a directory.
    file_type -- file type.
    
    return
    ------
    file       -- a log file to create a plot.
    """
    k = 0
    print('== Log List ==')
    for file in files:
        print("[" + str(k) + "]. " + file)
        k += 1
    print('Select a Log to display the plot')
    idx = int(input())
    return files[idx]

def main():
    # history log path
    log_path = '../plots/'
    # get the logs
    logs = os.listdir(log_path)
    # get and load the file
    log_name = select(logs)
    log = pd.read_csv(log_path + log_name, sep = ',')
    # plot
    plt.plot(log['epochs'].values, log['epoch_loss'].values)
    plt.plot(log['epochs'].values, log['training_loss'].values)
    plt.legend(['Epoch Loss', 'Training Loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Model Log')
    plt.show()

if __name__ == '__main__':
    main()