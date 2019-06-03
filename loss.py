#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:48:05 2019

@author: Orlando Ciricosta 
"""

from keras import backend as K
l_coord = 5.
l_noobj = 0.5



def yoloss(y_true, y_pred):
    
# calculate first the IOU tensors for the 2 boxes in each grid cell
    box1_pred =  y_pred[...,0:4]
    box2_pred =  y_pred[...,5:9]
    
    box_true = y_true[...,0:4]
        
    # Box 1
    x1b1 = K.maximum( box1_pred[...,0]-0.5*box1_pred[...,2] , box_true[...,0]-0.5*box_true[...,2] )
    y1b1 = K.maximum( box1_pred[...,1]-0.5*box1_pred[...,3] , box_true[...,1]-0.5*box_true[...,3] )
    x2b1 = K.minimum( box1_pred[...,0]+0.5*box1_pred[...,2] , box_true[...,0]+0.5*box_true[...,2] )
    y2b1 = K.minimum( box1_pred[...,1]+0.5*box1_pred[...,3] , box_true[...,1]+0.5*box_true[...,3] )
    
    intersection1 = K.maximum(x2b1-x1b1, 0)*K.maximum(y2b1-y1b1, 0)
    union1 = (box1_pred[...,2]*box1_pred[...,3] + 
                            box_true[...,2]*box_true[...,3] - 
                            intersection1 + K.epsilon() )
    iou1 = intersection1 / union1
    iou1 = K.expand_dims(iou1)
    
    # Box 2
    x1b2 = K.maximum( box2_pred[...,0]-0.5*box2_pred[...,2] , box_true[...,0]-0.5*box_true[...,2] )
    y1b2 = K.maximum( box2_pred[...,1]-0.5*box2_pred[...,3] , box_true[...,1]-0.5*box_true[...,3] )
    x2b2 = K.minimum( box2_pred[...,0]+0.5*box2_pred[...,2] , box_true[...,0]+0.5*box_true[...,2] )
    y2b2 = K.minimum( box2_pred[...,1]+0.5*box2_pred[...,3] , box_true[...,1]+0.5*box_true[...,3] )
    
    intersection2 = K.maximum(x2b2-x1b2, 0)*K.maximum(y2b2-y1b2, 0)
    union2 = (box2_pred[...,2]*box2_pred[...,3] + 
                            box_true[...,2]*box_true[...,3] - 
                            intersection2 + K.epsilon() )
    iou2 = intersection2 / union2
    iou2 = K.expand_dims(iou2)

# Get the maximum IOU --> which box is resposible for the prediction, plus the value of that IOU
    box_iou_max = K.expand_dims(
                        K.cast(
                            K.argmax( K.concatenate([iou1,iou2])),
                            y_pred.dtype
                        )
                    )
    # shape = (None,S,S,1), casted to a float to be able to multiply float tensors in the following
    
    
    IOU_max = K.maximum(iou1,iou2)

    
# Now build a revised version of y_true, y_pred, both containing only the box of maximum IOU,
# and with c=max_iou for y_pred
    
    ytrue = K.concatenate(
                [   y_true[..., 0:4],
                    IOU_max,
                    y_true[...,10:]
                ]
            )
    
    ypred = K.concatenate(
                [   y_pred[..., 0:5] * (1-box_iou_max) + y_pred[..., 5:10] * box_iou_max, 
                    y_pred[...,10:]
                ]
            )
    
# The last needed tensor is the 1_i tensor = 1 if an object is in the grid cell
    One = K.max(y_true[..., 10:], axis=-1)
    # shape = (None,S,S) as it is mainly multiplied by particular elements of shape ytrue[...,i].shape
    # will use K.expand_dims for the last term of the loss where tensors have shape ytrue[...,i:j].shape

# Finally it is time to build the loss function:

    loss = (l_coord * K.sum( One * ( K.square( ypred[...,0] - ytrue[...,0] ) +
                                     K.square( ypred[...,1] - ytrue[...,1] ) ) 
                          ) +
            l_coord * K.sum( One * ( K.square( K.sqrt(ypred[...,2]) - K.sqrt(ytrue[...,2]) ) +
                                     K.square( K.sqrt(ypred[...,3]) - K.sqrt(ytrue[...,3]) ) ) 
                          ) +
            K.sum( One * ( K.square( ypred[...,4] - ytrue[...,4] ) ) ) +
            l_noobj * K.sum( (1. - One) * ( K.square( ypred[...,4] - ytrue[...,4] ) ) ) +
            K.sum( K.expand_dims(One) *  ( K.square( ypred[...,10:] - ytrue[...,10:] ) ) )
           )
    
    return loss