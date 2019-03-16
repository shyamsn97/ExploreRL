import numpy as np
import tensorflow as tf
import torch

def create_linear_tf(input_dims,output_dims,batch_norm=False,regularizer_weight=0.0):
	layers = []
	lin = tf.keras.layers.Dense(input_shape=(input_dims,),units=output_dims, 
				bias_regularizer=tf.keras.regularizers.l2(regularizer_weight))
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