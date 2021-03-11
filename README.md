# ML-course-MNIST

This project made to teach the basics of machine learning on the benchmark dataset MNIST. 

MNIST has 60 000 training images and 10 000 test images. The images are of handwritten digits (0-9). 

## Install
To get started you will need to have Python 3.6 or newer. 
You can get it here https://www.python.org/

For this project we will need numpy and tensorflow 2.x . 
Intall the packages:
``` pip install numpy tensorflow ```

## Get the code
I have provided some code in src/main to get you started with machine learning on the MNIST dataset. 
To get the code you can either download the file from github directly, or clone the git repository
``` git clone https://github.com/aridalnorkart/ML-course-MNIST.git```

## Run the code
The provided code should run without any modifications. 
To run it navigate to the folder where you placed main.py run ```python main.py```.

When the code is executed the first time Keras will automaticly download the MNIST dataset and store it on your computer. 
The script will make the dataset ready for training a convolutional neural network. 
The network is defined in the code and will be trained on 54000 images of handwritten digits. (6000 images will be used as a validation set)

When the model is finished training it will be used to make predictions on a test set of 10000 images.
This gives you a test accuracy that tells you how well your model would be on new data that it did not see during training. 

## Modify the code 
To get a higher test accuracy you will have to modify the code. Read the comments for suggestions for how to modify it.
After you have made the modifications, train and test the model again. 

