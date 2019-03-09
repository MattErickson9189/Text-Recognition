import tensorflow as tf
import numpy as np
import os



class NeuralNet:
    #This is a class that will have methods for setting up the CNN and RNN

    height = 128
    width = 32

    def __init__(self, batchSize, learningRate, trainSessions, wordList):

        self.batchSize = batchSize
        self.learningRate = learningRate
        self.trainSession = trainSessions
        self.list = wordList

        self.CNN(self)

        self.numTrained = 0
        self.learningRate = tf.placeholder(tf.float, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer= tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)




    def CNN(self):

        cnnIn = tf.expand_dims(input=self.inputImgs, axis=3)

        kValues = [5,5,3,3]
        featureVals = [1,32,64,128,128,256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        pool = cnnIn

        for i in range(numLayers):

            kernel = tf.Variable(tf.truncated_normal([kValues[i], kValues[i], featureVals[i], featureVals[i+1]], stddev=0.1))
            conv = tf.nn.conv2d(pool,kernel, padding="SAME", strides=(1,1,1,1))
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1],1), "VAILD")

        self.cnnOut4d = pool



    def trainBatch(self, batch):
        