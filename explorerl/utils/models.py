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

class FeedForwardNN(tf.keras.Model):
    def __init__(self,input_space,output_space,batch_norm=False):
        super(FeedForwardNN, self).__init__()
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
