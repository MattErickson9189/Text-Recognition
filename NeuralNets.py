import tensorflow as tf
import numpy as np
import os



class NeuralNet:
    #This is a class that will have methods for setting up the CNN and RNN

    def __init__(self, batchSize, learningRate, trainSessions):

        self.batchSize = batchSize
        self.learningRate = learningRate
        self.trainSession = trainSessions

        n_classes = 26

        x = tf.placeholders("float", [None, 128,32,1])
        y = tf.placeholders("float", [None, n_classes])


    def conv2d(self,x,w,b,strides =1):
        #Wrapped for bias and relu activation

        x = tf.nn.conv2d(x,w,strides = [1,strides,strides,1], padding = "SAME")
        y = tf.nn.bias_add(x,b)
        return tf.nn.relu(x)

    def maxpool2D(selfx, k=2):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1],padding="SAME")
    