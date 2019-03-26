import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

###################### Tensorflow ######################
def create_linear_tf(input_dims,output_dims,batch_norm=False,regularizer_weight=0.001):
    layers = []
    lin = tf.keras.layers.Dense(input_shape=(input_dims,),units=output_dims,
                                bias_regularizer=tf.keras.regularizers.l2(regularizer_weight),dtype='float32')
    layers.append(lin)
    if batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
        return tf.keras.Sequential(layers)

###################### Torch ######################
def create_linear_torch(input_dims,output_dims,batch_norm=False):
    layers = []
    lin = torch.nn.Linear(input_dims,output_dims)
    layers.append(lin)
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(output_dims))
    return torch.nn.Sequential(*layers)

def conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, 
           batch_norm=True, init_zero_weights=False,w_max=False,relu=False):
    """
        Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(out_channels))
    if relu:
        layers.append(torch.nn.ReLU())
    if w_max:
        layers.append(torch.nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size))
    return torch.nn.Sequential(*layers)

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, 
           batch_norm=True, init_zero_weights=False,w_max=False,relu=False):
    """
        Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)
    if batch_norm:
        layers.append(torch.nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(torch.nn.ReLU())
    if w_max:
        layers.append(torch.nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size))
    return torch.nn.Sequential(*layers)