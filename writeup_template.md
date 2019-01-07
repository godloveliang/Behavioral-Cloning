# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/origin.png "Normal Image"
[image2]: ./examples/flip.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64. 
The model includes RELU layers to introduce nonlinearity (code line 66,68,70), and the data is normalized in the model using a Keras lambda layer (code line 65). 

TO reduce model complexity and reduce learning time, I added two maxpool layers.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 74, 76). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

I used udacity data set to train the model.In order to make full use of the data, I not only used the middle camera photos, but also used the photos of the left and right cameras. When using the photos of the left and right cameras to train the steering angle, I will give a certain correction factor to it.

In order to further increase the amount of data, I used a flip. In this way, not only the amount of training data is doubled, but the training data also contains information in the opposite direction.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use 3 layer 3x3 convolutional layer and 4 fuuly connected layer. I think this model simple and efficient, but when run the simulator to see how well the car was driving around track one, It performs very poorly, only a few steps he vehicle fell off the track.

From the first experience, I think the model is too simple. So I add one more convolutional layer. This time the vichle performance has improved a lot, but at the big corner the vehicle fell off the track.

Then I try the nvida model, I thought this model might be appropriate because this model is used for training real car.
I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Then I add two dropout layers affter fully connected layer, this may  handle the overfitting well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

* Image cropping
* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Fully connected: neurons: 100, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 50, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)

#### 3. Creation of the Training Set & Training Process

Due to the serious delay of the simulator, it is difficult for me to drive a car to collect data. So I used the data set provided by Udacity.
To augment the data sat, I flipped images ,this could double the data set. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

To make most use of the data set, I not only use the center images but also the left and right imags. In order to be able to use the left and right images, correction factors need to add.

After the above process, I had 48216 number of data points. I finally randomly shuffled the data set and put 10% of the data into a validation set. 

Before train
* Image cropping, cut off the top and buttom unnecessary pixel, avoid confuse the model.
* Image normalization

As for train
* I used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.
* I used Adam optimizer for optimization.
