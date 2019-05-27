import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as F

#Tensorflow
class DualPolicyValueHead(tf.keras.Model):
    def __init__(self,input_space,output_space,configs={"softmax"}):
        super(DualPolicyValueHead, self).__init__()
        
class LinearEstimatorTf(tf.keras.Model):
    def __init__(self,input_space,output_space,configs={"softmax"}):
        super(LinearEstimatorTf, self).__init__()
        self.dense = tf.keras.layers.Dense(units=output_space,input_shape=input_space, bias_regularizer=tf.keras.regularizers.l2(0.0001),dtype='float32')
        if "softmax" in configs:
            self.dense = tf.keras.models.Sequential([self.dense,tf.keras.layers.Softmax()])
    def call(self,x,training=True):
        x = self.dense(x)
        return x

class FeedForwardNNTf(tf.keras.Model):
    def __init__(self,input_space,output_space,configs={"softmax"}):
        super(FeedForwardNNTf, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=24,input_shape=input_space, bias_regularizer=tf.keras.regularizers.l2(0.0001),dtype='float32')
        self.dense2 = tf.keras.layers.Dense(units=24, bias_regularizer=tf.keras.regularizers.l2(0.0001),dtype='float32')
        self.dense3 = tf.keras.layers.Dense(units=output_space,bias_regularizer=tf.keras.regularizers.l2(0.0001),dtype='float32')
        if "softmax" in configs:
            self.dense3 = tf.keras.models.Sequential([self.dense3,tf.keras.layers.Softmax()])

    def call(self,x,training=True):
        x = tf.keras.activations.relu(self.dense1(x))
        x = tf.keras.activations.relu(self.dense2(x))
        x = self.dense3(x)
        return x

#Pytorch
class LinearEstimatorTorch(torch.nn.Module):
    def __init__(self,input_space,output_space,configs={"softmax"}):
        super(LinearEstimatorTorch,self).__init__()
        self.linear = torch.nn.Linear(*input_space,output_space)
        if "softmax" in configs:
            self.linear = torch.nn.Sequential(*[self.linear,torch.nn.Softmax(dim=-1)])
    def forward(self,x):
        x = self.linear(x)
        return x



