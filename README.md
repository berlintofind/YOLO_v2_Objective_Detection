# YOLO_v2_Objective_Detection
Objective Detection models based on YOLO_V2 algorithm, Keras, tensorflow backend

By Tianyang Zhao

## Table of Contents
1. Introduction
2. Models
3. Results
4. Reference

## Introduction
This repository contains the Deep Residual Network model. The goal of the model is to decipher sign language (from 0 -> 5) .
Here are examples for each number, and how an explanation of representing the labels. These are the original pictures, before lowering the image resolutoion to 64 by 64 pixels. 

![image](https://github.com/berlintofind/Multilayer-Perceptron/blob/main/images/hands.png)

#### Note 
Note that this is a subset of the SIGNS dataset. The complete dataset contains many more signs.

#### Packages Version
1. Python 3.6.9 
2. tensorflow-gpu 1.13.1
3. numpy 1.19.1
4. CUDA Version 10.1
5. scipy 1.2.1
6. keras-gpu 2.3.1


## Models

The model you've just run is actually able to detect 80 different classes listed in "coco_classes.txt". To test the model on your own images，





Deep "plain" convolutional neural networks don't work in practice because they are hard to train due to gradients exploration. ResNet block with skip connections can help to address this problem thus to make the networks go deeper. Two types of blocks are implemented in this model: the identity block and the convolutional block. Deep Residual Networks are built by stacking these blocks together.

The model contains 50 hidden layers. Using a softmax output layer, the models is able to generalizes more than two classes of outputs.

The architecture of the model is shown as follows:
![image](https://github.com/berlintofind/Image-classification-models-with-Deep-Residual-Networks/blob/main/images/resnet_kiank.png)

The input image has a size of (64,64,3). 

For purpose of demonstration, the model is trained for 2 epoches firstly. The model requires several hours of training with the ResNet. So the model will use the pre-trained model instead which is saved as *ResNet50.h5* file. In this way, lots of time is saved. The result of the prediction is saved in the *Result.CSV* file.

You can try to put your image inside for prediction. To test your own sign image, only need to change the file name in line 239.
In the last, the whole structure of the ResNet is saved in Scalable Vector Graphics (SVG) format as *model.png*.


Keep safe and see you soon!

## Results

**Result of prediction of test.jpg** :

Found 4 boxes for IMG_5753.jpg
person 0.72 (302, 1966) (649, 2907)
person 0.77 (1824, 1839) (2058, 2694)
person 0.79 (2940, 1816) (3176, 2521)
person 0.82 (2195, 1793) (2393, 2422)

**Result of prediction of Checkpoint Charlie.jpg** :
![image](https://github.com/berlintofind/YOLO_v2_Objective_Detection/blob/master/out/Checkpoint%20Charlie.jpg)
Found 3 boxes for Checkpoint Charlie.jpg
person 0.78 (1050, 648) (1137, 902)
person 0.78 (651, 657) (738, 958)
person 0.79 (783, 642) (856, 863)

**Result of prediction of The_Nightwatch_by_Rembrandt_-_Rijksmuseum.jpg** :
![image](https://github.com/berlintofind/YOLO_v2_Objective_Detection/blob/master/out/The_Nightwatch_by_Rembrandt_-_Rijksmuseum.jpg)
Found 3 boxes for The_Nightwatch_by_Rembrandt_-_Rijksmuseum.jpg
person 0.74 (5250, 5639) (7696, 11528)
person 0.77 (1826, 6052) (3740, 11528)
person 0.84 (7567, 6093) (9934, 11528)

## Reference
1. [Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)
2. [Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)
3. [Allan Zelener - YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
4. [The official YOLO website (https://pjreddie.com/darknet/yolo/)](https://pjreddie.com/darknet/yolo/)
