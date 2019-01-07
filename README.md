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






To avoid overfitting, a set of dropout layers were applied to the fully connected section of the model, with a probability of 70%.


There was a total of five defined classes, four of them represent the basic actions that the model can take on the game, this actions are: turn to the left, turn to the right, accelerate and return. The last class indicates when the race is over, and is used to stop the main script. These classes can be visualized in the figure below:
