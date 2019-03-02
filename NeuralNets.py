import tensorflow as tf
import numpy as np
import os



class NeuralNet:
    #This is a class that will have methods for setting up the CNN and RNN

    def __init__(self, batchSize, learningRate, trainSessions):

        self.batchSize = batchSize
        self.learningRate = learningRate
        self.trainSession = trainSessions
        