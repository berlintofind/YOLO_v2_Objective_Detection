# YOLO_v2_Objective_Detection
Objective Detection model based on YOLO_V2 algorithm, Keras, tensorflow backend

By Tianyang Zhao

## Table of Contents
1. Introduction
2. Models
3. Results
4. Reference

## Introduction
This repository contains the Objective Detection model. The goal of the model is to detect and localize various objects well-defined.

#### Packages Version
1. Python 3.6.9 
2. tensorflow-gpu 1.13.1
3. numpy 1.19.1
4. CUDA Version 10.1
5. scipy 1.2.1
6. keras-gpu 2.3.1


## Models

The model is actually able to detect 80 different classes listed in "coco_classes.txt". 

It should be noted that the model load the pre-trained Keras model of YOLO_V2 which saved as "yolo.h5". These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this document. The user could also train the model based on his own datasets. However, training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes.

The input images of the model will be resize to (608,608,3), and then scale up to its origin size as outputs. Red boxes are used to show the positions of objects detected and the possibilities. The maximum numbers of boxes allowed is 13, and the threshold to judge if the model has enough confindence on its result is 0.7. Objects deteced with chance lower than the level will be discarded. iou_threshold is to decide the threshold of overlapping ratio. Objects with higher ratio of overlapping are more likely to be the same object. User can adjust these parameters by himself.

Three different types of images are tested in this model for prediction.
The first image "test.jpg" contains cars, has a size of (720., 1280.), 300KB

The second image "Checkpoint Charlie.jpg" taken from Checkpoint Charlie, Berlin, has a size of (1080.,1440.), 227KB

The third image "The_Nightwatch_by_Rembrandt_Rijksmuseum.jpg", is the famous painting by one of my favourite painters, Rembrandt. Download form Google, has a size of 34.9MB, (12528.,14168.)

To test the model on your own images，the use only need change the name with his own image in line 189 and the size of the image in both line 100 and 141 of Object_Detection.py.

#### Note

Due to the size of the third image, it may take some time to render. The size will increase to 48.2MB.


Keep safe and see you soon!

## Results

**Result of prediction of test.jpg** :
![image](https://github.com/berlintofind/YOLO_v2_Objective_Detection/blob/master/out/test.jpg)

Found 3 boxes for test.jpg
car 0.74 (159, 303) (346, 440)
car 0.80 (761, 282) (942, 412)
car 0.89 (367, 300) (745, 648)

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
4. [The official YOLO website](https://pjreddie.com/darknet/yolo/)
