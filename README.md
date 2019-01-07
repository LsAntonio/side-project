# Self-KartDriving



# Requirements:
To train and running this project, you can use Anaconda 64-Bit with python 3.7. Then, create a virtual environment and install the following:

* pillow=5.3.0
* numpy=1.15.4
* pytorch-cpu=0.4.1
* torchvision-cpu=0.2.1
* mss=3.3.2
* pynput=1.4

Note that this project was only tested with Ubuntu 16.04 LTS. Therefore it may not work with other systems.

# Model Architecture:
The model is composed by an alexnet model, which was fine-tuned. The model receives an input with shape: 224 x 224 x 3, where the two first elements represent the width and hight, and the last one, the number of channels. Also, the last part of the original model has been replaced with a set of fully connected layers. A representation of the model can be visualized in the figure below:
![Model Architecture](figure_1.png)

To avoid overfitting, a set of dropout layers were applied to the fully connected section of the model, with a probability of 70%. There was a total of five defined classes, four of them represent the basic actions that the model can take on the game, this actions are: turn to the left, turn to the right, accelerate and return. The last class indicates when the race is over, and is used to stop the main script. These classes can be visualized in the figure below:
![Model Classes](figure_2.png)

# Training:
In order to train the model, a set of images with different time intervals where extracted from a gameplay video of the __CandelaCity__ track. Then, each image was labeled accordingly with their respective classes. These are stored in a folder called dataset.

To train this model from scratch, you have two options: 
* Download the dataset from [here](https://drive.google.com/open?id=1W7DgjqPx3PZkEdPDnsxzkKieilURbwv0). Once download, unzip it in the same level as the script folder. You must now have a folder called dataset with a folder inside called train. To add more data, you need to capture it from video/images. Here you can use PIL, or another library, just make sure you to capture the images without the top part (as shown in the __Figure 2__).
* Generate your own data. In this case, you can choose the image ratio which you like. However, you must gather all the data from zero.

In both cases, make sure the model is getting the right image. To check that, go the the ```run.py``` file and modify the values from the mss screenshot until you get an adequate image:
```
# Modify these values if they do not fit your screen.
if screen == 'r':
  # take a right screenshot
  monitor = {"top": 145, "left": 342, "width": 1018, "height": 620}
elif screen == 'l':
  # take a left screenshot
  monitor = {"top": 145, "left": 70, "width": 1018, "height": 620}
````
Now, you can use the ```training.py``` file. It would require to input the number of epochs to train (make sure you have enough free space in your hard drive). Also, the model is defined in the ```model_helper.py``` file. There you can change the model’s parameters like:  batch size, learning rate, optimizer, etc.

You can also check the model(s) performance with the ```eval_model.py``` file. Finally, you can display a plot of the loss of the entire training with the ```plot.py``` file.

Here is an example of using all the training scripts:
<p align="center">
<a href="https://youtu.be/Vl6mrSCoWhg" target="_blank">
  <img src="https://drive.google.com/uc?export=view&id=1RziMx0sFIQ9qLLZZLnsyy-M5KA-ZULZL" alt="Training" width = "500", height = "250">
</a>
</p>
