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
from PIL import Image
import torch
import time 
import mss
import mss.tools
from pynput.keyboard import Key, Controller
from model_helper import load_model, preprocess, select

np.random.seed(5)

####################
# Helper Functions #
####################

def main():
    # models path
    model_path = '../models/'
    #classes
    classes = ['back', 'done', 'left', 'right', 'up']
    print('############## Settings ##############')
    screen = str(input('Game position configuration [(l)eft|(r)ight]: '))
    # set the number of steps to run
    steps = int(input('Number of steps to run: '))
    # check the game status
    done = 0
    # select a model
    model_list = list(os.listdir(model_path))
    # load the model
    model = load_model(select(model_list, 'Model'))
    # wait until focus the game as current window
    wait = int(input('Number of seconds to wait, before begin: '))
    time.sleep(wait)

    # move model to the cpu
    model.to('cpu')
    # set the model to evaluation mode
    model.eval()
    # set the keyboard
    keyboard = Controller()
    # main loop
    for step in range(steps):
        with mss.mss() as sct:
            # Modify these values if they do not fit your screen.
            if screen == 'r':
                # take a right screenshot
                monitor = {"top": 145, "left": 342, "width": 1018, "height": 620}
            elif screen == 'l':
                # take a left screenshot
                monitor = {"top": 145, "left": 70, "width": 1018, "height": 620}
            else:
                print("Monitor can't be found!.")
                exit()
            # grab the screenshot data
            sct_img = sct.grab(monitor)
            # convert the raw to PIL
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # preprocess the image
        image = preprocess(img)
        # set image into a single batch tensor
        image = image.float()
        image = image.unsqueeze(0)
        # get predictions
        output = model(image)
        # convert output probabilities to predicted class
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        # print the predicted class
        print('class: {}'.format(classes[top_class]))
        plus = np.random.uniform(0, .2)
        # turn to the left
        if classes[top_class] == 'left':
            keyboard.press(Key.left)
            keyboard.press(Key.up)
            time.sleep(0.3 + plus)
            keyboard.release(Key.left)
            keyboard.release(Key.up)
        # turn to the right
        if classes[top_class] == 'right':
            keyboard.press(Key.right)
            keyboard.press(Key.up)
            time.sleep(0.3 + plus) 
            keyboard.release(Key.right)
            keyboard.release(Key.up)
        # accelerate
        if classes[top_class] == 'up':
            keyboard.press(Key.up)
            time.sleep(1.0)
            keyboard.release(Key.up)
        # get back to the track
        if classes[top_class] == 'back':
            keyboard.press(Key.backspace)
            time.sleep(0.2)
            keyboard.release(Key.backspace)
            time.sleep(2)
        # update the game status
        if classes[top_class] == 'done':
            done += 1
            
        # terminate the script
        if done >= 10:
            print('Done!...')
            break;

if __name__ == '__main__':
    main()
