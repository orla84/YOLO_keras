#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:52:43 2019

@author: Orlando Ciricosta 
"""

from keras.layers import Dense, Conv2D, LeakyReLU, MaxPool2D
from keras.layers import InputLayer, Reshape, Flatten
from keras.models import Sequential

yolo = Sequential()

yolo.add(InputLayer(input_shape = (448,448,3)))

yolo.add(Conv2D(64,7, strides=2, padding='same', name='conv-0_block-0'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-0'))
yolo.add(MaxPool2D(strides=2, name='pool_block-0'))

yolo.add(Conv2D(192,3, padding='same', name='conv-0_block-1'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-1'))
yolo.add(MaxPool2D(strides=2, name='pool_block-1'))

yolo.add(Conv2D(128,1, padding='same', name='conv-0_block-2'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-2'))
yolo.add(Conv2D(256,3, padding='same', name='conv-1_block-2'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-1_block-2'))
yolo.add(Conv2D(256,1, padding='same', name='conv-2_block-2'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-2_block-2'))
yolo.add(Conv2D(512,3, padding='same', name='conv-3_block-2'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-3_block-2'))
yolo.add(MaxPool2D(strides=2, name='pool_block-2'))

yolo.add(Conv2D(256,1, padding='same', name='conv-0_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-3'))
yolo.add(Conv2D(512,3, padding='same', name='conv-1_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-1_block-3'))
yolo.add(Conv2D(256,1, padding='same', name='conv-2_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-2_block-3'))
yolo.add(Conv2D(512,3, padding='same', name='conv-3_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-3_block-3'))
yolo.add(Conv2D(256,1, padding='same', name='conv-4_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-4_block-3'))
yolo.add(Conv2D(512,3, padding='same', name='conv-5_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-5_block-3'))
yolo.add(Conv2D(256,1, padding='same', name='conv-6_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-6_block-3'))
yolo.add(Conv2D(512,3, padding='same', name='conv-7_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-7_block-3'))
yolo.add(Conv2D(512,1, padding='same', name='conv-8_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-8_block-3'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-9_block-3'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-9_block-3'))
yolo.add(MaxPool2D(strides=2, name='pool_block-3'))

yolo.add(Conv2D(512,1, padding='same', name='conv-0_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-4'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-1_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-1_block-4'))
yolo.add(Conv2D(512,1, padding='same', name='conv-2_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-2_block-4'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-3_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-3_block-4'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-4_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-4_block-4'))
yolo.add(Conv2D(1024,3, strides=2, padding='same', name='conv-5_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-5_block-4'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-6_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-6_block-4'))
yolo.add(Conv2D(1024,3, padding='same', name='conv-7_block-4'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-7_block-4'))

yolo.add(Flatten(name='flatten'))
yolo.add(Dense(4096, name='dense_0'))
yolo.add(LeakyReLU(alpha=0.1, name='leaky-0_block-5'))
yolo.add(Dense(1470, name='dense_1'))
yolo.add(Reshape((7,7,30), name='reshape_out'))

yolo.summary()