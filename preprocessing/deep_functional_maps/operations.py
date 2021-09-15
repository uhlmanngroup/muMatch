## Standard Lib Imports ##
import os
import sys
import math

import numpy as np

## tensorflow ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf


''' ========================================================================================== '''
'''                                         Beginning                                          '''
''' ========================================================================================== '''


@tf.function
def solve_lstsq(A, B):
    At  =  tf.transpose(A, [0,2,1])
    Bt  =  tf.transpose(B, [0,2,1])
    Ct  =  tf.linalg.lstsq(At,Bt, l2_regularizer=1e-3)
    C   =  tf.transpose(Ct, [0,2,1])
    return C


''' ========================================================================================== '''
'''                                    Ensemble Pair Loss                                      '''
''' ========================================================================================== '''

def correspondenceMatrix(sigs, evecs_t):
    A,B = (tf.matmul(e,s) for (e,s) in zip(evecs_t,sigs))
    C = solve_lstsq(A,B)
    return C


def softCorrespondenceEnsemble(C, evecs_1_t, evecs_2):
    P = tf.linalg.matmul(tf.linalg.matmul(evecs_2, C), evecs_1_t)
    P = tf.math.l2_normalize(P, axis = 1, epsilon=1e-6)
    P = tf.transpose(P, perm = [0,2,1])
    return tf.pow(P,2)


def geodesicErrorEnsemble(P, dist_1, dist_2):
    dist_21 = tf.linalg.matmul(tf.linalg.matmul(P, dist_2), P, transpose_b = True)
    unsupervised_loss = tf.nn.l2_loss(dist_21 - dist_1)
    unsupervised_loss /= tf.cast(tf.shape(P)[0] * tf.shape(P)[1] * tf.shape(P)[1], unsupervised_loss.dtype)
    return unsupervised_loss

''' ========================================================================================== '''
'''                               Functional Mapping Network                                   '''
''' ========================================================================================== '''


class residualLayer(tf.keras.Model):

    def __init__(self, dim, trainable):
        super(residualLayer,self).__init__()
        self.dim    = dim
        self.dense1 = tf.keras.layers.Dense(dim, activation = None)
        self.batch1 = tf.keras.layers.BatchNormalization(center=True, scale=True, trainable=trainable)
        self.dense2 = tf.keras.layers.Dense(dim, activation = None)
        self.batch2 = tf.keras.layers.BatchNormalization(center=True, scale=True, trainable=trainable)

    def __call__(self, x):
        assert(self.dim == x.shape[-1])
        y = self.dense1(x)
        y = self.batch1(y)
        y = tf.nn.relu(y)
        y = self.dense2(y)
        y = self.batch2(y)
        y += x
        return tf.nn.relu(y)


class residualNet(tf.keras.Model):

    def __init__(self, num_layers, num_descriptors, training):
        super(residualNet, self).__init__()
        self.__layers = [residualLayer(num_descriptors,training) for _ in range(num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




''' ========================================================================================== '''
'''                                           End                                              '''
''' ========================================================================================== '''


if __name__ == "__main__":
	mod = resNet(5, 3000, 200)
	mod.summary()
