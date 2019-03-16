import numpy as np
import torch
import tensorflow as tf

#Tensorflow
class LinearEstimatorTF(tf.keras.Model):
    def __init__(self,input_space,output_space):
        super(LinearEstimatorTF, self).__init__()
        self.dense = tf.keras.layers.Dense(output_space,input_shape=(input_space,), bias_regularizer=tf.keras.regularizers.l2(0.0001))
    def call(self,x,training=True):
        x = self.dense(x)
        return x

#Pytorch
class LinearEstimatorTorch(torch.nn.Module):
    def __init__(self,input_space,output_space):
        super(LinearEstimatorTorch,self).__init__()
        self.linear = torch.nn.Linear(input_space,output_space)
    
    def forward(self,x):
        x = self.linear(x)
        return x
