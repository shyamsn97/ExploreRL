import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as F

#Tensorflow
class LinearEstimatorTF(tf.keras.Model):
    def __init__(self,input_space,output_space):
        super(LinearEstimatorTF, self).__init__()
        self.dense = tf.keras.layers.Dense(output_space,input_shape=(input_space,), bias_regularizer=tf.keras.regularizers.l2(0.0001))
    def call(self,x,training=True):
        x = self.dense(x)
        return x

class FeedForwardNNTF(tf.keras.Model):
    def __init__(self,input_space,output_space,batch_norm=False):
        super(FeedForwardNNTF, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=24,input_shape=(input_space,))
        self.dense2 = tf.keras.layers.Dense(units=24)
        self.dense3 = tf.keras.layers.Dense(units=output_space)

    def call(self,x,training=True):
        x = tf.nn.relu(self.dense1(x))
        x = tf.nn.relu(self.dense2(x))
        x = self.dense3(x)
        return x

#Pytorch
class LinearEstimatorTorch(torch.nn.Module):
    def __init__(self,input_space,output_space):
        super(LinearEstimatorTorch,self).__init__()
        self.linear = torch.nn.Linear(input_space,output_space)
    
    def forward(self,x):
        x = self.linear(x)
        return x

class FeedForwardNNTorch(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(FeedForwardNNTorch, self).__init__()
        self.linear1 = torch.nn.Linear(input_size,24)
        self.linear2 = torch.nn.Linear(24,24)
        self.linear3 = torch.nn.Linear(24,output_size)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

