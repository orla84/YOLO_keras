# YOLO_keras
Keras implementation of YOLO (original paper)


### YOLO version

Citing from the [paper](https://arxiv.org/pdf/1506.02640.pdf):

``Our system models detection as a regression problem. It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes, and C class probabilities. These predictions are encoded as an
S × S × (B ∗ 5 + C) tensor.
For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor.``

The parameters S,B,C there described are assumed in the network architecture used in this repo. Generalization efforts have been kept at minimum -- the code is meant to be used as a template to understand the Yolo model, rather than as a production-ready code.
The input image is assumed RGB, of dimensions 448x448


### Structure of the labels/predictions
I assume that the (7,7,30)-shaped labels are built as follows:

`label[i,j,:]` = labels for the i,j grid cell of the image

`label[i,j,0:5]` = xc,yc,w,h,c for the first of the two boxes detector (5:10 for the second)

`label[i,j,10:30]` = P(class|object) that the i,j subsection contains an object from one of the 20 classes