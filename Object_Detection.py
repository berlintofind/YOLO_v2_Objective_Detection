# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:25:15 2020

@author: To find Berlin
"""
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

# 要把 19*19*5*85 的结果转化的更易于使用NMS,要先设置一个阈值过滤 -- 将结果转化的易于使用过滤
# The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It is convenient to rearrange 
# the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:

# box_confidence: tensor of shape  (19×19,5,1)(19×19,5,1)  containing  pcpc  (confidence probability that there's some object) 
# for each of the 5 boxes predicted in each of the 19x19 cells.
# boxes: tensor of shape  (19×19,5,4)(19×19,5,4)  containing the midpoint and dimensions  (bx,by,bh,bw)(bx,by,bh,bw)  
# for each of the 5 boxes in each cell.
# box_class_probs: tensor of shape  (19×19,5,80)(19×19,5,80)  containing the "class probabilities"  (c1,c2,...c80)(c1,c2,...c80)  
# for each of the 80 classes for each of the 5 boxes per cell

# Keras argmax
# Keras max
# keras.backend.argmax 和 Keras.argmax 什么区别？

# 筛选阈值本身这步，用mask就可以
# For the tf.boolean_mask, we can keep the default axis=None
# Reminder: to call a Keras function, you should use K.function(...).
# keras.backend 和 use tf as backend什么区别

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = (box_class_scores >= threshold)
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores, boxes, classes
with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

# 错误1： ,axis=-1 K.max 应该这样用, 不然会折叠成 一个sclar
# 在 utils.py 怎么做到的
# Note In the test for yolo_filter_boxes, we're using random numbers to test the function. 
# In real data, the box_class_probs would contain non-zero values between 0 and 1 for the probabilities. 
# The box coordinates in boxes would also be chosen so that lengths and heights are non-negative.


# YOLO non-max suppression
# The key steps are:
# Select the box that has the highest score.
# Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= iou_threshold).
# Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.


# tf.image.non_max_suppression()
# K.gather()
# K.variable()
# K.get_session()


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')          # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores,nms_indices )
    boxes = K.gather(boxes,nms_indices )
    classes = K.gather(classes,nms_indices )
    
    return scores, boxes, classes
with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

# 但是这个算法怎么和实际结合？
# yolo_outputs 这一项是怎么经过CNN 得到的
# 哪里是重复检测，下一个非重叠的概率最高object的？

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5)
    
    return scores, boxes, classes

with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

# 这个算法的实施，只是实际过程的很小部分，是全部CNN 调参完成之后的判定
# 没有体现哪里有重载 pre-training 的参数  -- 这个Predict 的过程，怎么经历的？
# 没有体现反向传播的训练过程
# cost function 也没有定义
# 没有图片原始输入处理的过程

# 像素不同的时候，是怎么处理的？


# Summary for YOLO:
# Input image (608, 608, 3)
# The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output

# After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
# Each cell in a 19x19 grid over the input image gives 425 numbers.
# 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
# 85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and 80 is the number of classes we'd like to detect

# You then select only few boxes based on:
# Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
# Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

# This gives you YOLO's final output.


# 3. Test YOLO pre-trained model on images
# use a pre-trained model and test it on the car detection dataset. 
# We'll need a session to execute the computation graph and evaluate the tensors.
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg

# 什么是coco 模型？
# tensorflow_backend/ Kersa_backend 区别是什么






sess = K.get_session()

# 3.1 - Defining classes, anchors and image shape
# read_classes / read_anchors() 这些函数 怎么定义的
# image_shape 有没有别的 size 可以？ 比如手机像素

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)   

# 3.2 - Loading a pre-trained model
# 需要下载YOLO的预训练权重，为什么有194M, 预训练的CNN 模型是什么样的
# -- 是一个 22 层的CNN模型， 基本结构是： Conv - BN - Leaky Relu - max pooling - Conv
# Total params: 50,983,561
# Trainable params: 50,962,889
# Non-trainable params: 20,672

# 中间有concatenate 层 做什么用？
# 是深层网络，但是 似乎没有用到 resNet?
# 需要看YOLO 22 层的解析
# 为什么YOLO 模型中不使用 L2正则

# 去看github 的相关project

yolo_model = load_model("model_data/yolo.h5")  
# 这个是最重要的
yolo_model.summary()

# 为什么 K.summary()

# Reminder: this model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) 
# into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2)
# 22 层 CNN 的作用

# The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion.
# 然而这个不平凡变换是怎么做的
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# added yolo_outputs to your graph. 
# This set of 4 tensors is ready to be used as input by your yolo_eval function.
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# created a graph that can be summarized as follows:
# 1. yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output
# 2. yolo_model.output is processed by yolo_head. It gives you yolo_outputs
# 3. yolo_outputs goes through a filtering function, yolo_eval. It outputs your predictions: scores, boxes, classes

# 如果size 不是 608*608 ？
# sess 可以作为一个 参数传入？
def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores,boxes, classes],feed_dict={yolo_model.input: image_data ,K.learning_phase():0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
# Outputs: drawbox里有输出要求
# Found 10 boxes for IMG_5753.jpg
# person 0.45 (287, 447) (353, 577)
# person 0.53 (192, 463) (253, 611)
# person 0.60 (1076, 443) (1183, 623)
# person 0.62 (317, 449) (382, 577)
# person 0.65 (805, 438) (856, 519)
# person 0.67 (516, 439) (592, 629)
# person 0.72 (96, 468) (206, 692)
# person 0.77 (579, 438) (653, 641)
# person 0.79 (933, 432) (1008, 600)
# person 0.82 (697, 427) (760, 577)


# out_scores, out_boxes, out_classes = predict(sess, "The_Nightwatch_by_Rembrandt_-_Rijksmuseum.jpg")
# 这张图的 size就不可以，是1.82 倍了
# "IMG_5753.jpg"也不行, 因为预设尺寸不合适，所以，矩形框的显示并不能对准
# 可以检测出来
# 但只能检测出7个
# 应该 有15 - 18 个人显示
# 可能有很多原因参与：
# 1. IOU 的范围 2. 初始阈值的限制 3. CNN 本身还需要训练参数


# /opt/conda/lib/python3.6/site-packages/PIL/Image.py:2274: DecompressionBombWarning: Image size (163328704 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
#   DecompressionBombWarning)

 # The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. 
 # The pre-trained weights used in this exercise came from the official YOLO website.





