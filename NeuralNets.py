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

    def CTC(self):
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]),
                                       tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                           ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput,
                                             sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                                                         beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.list)
            wordChars = open('../Data/Lists/wordCharList.txt').read().splitlines()[0]
            corpus = open('../Data/Lists/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words',
                                                                    0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                                                    wordChars.encode('utf8'))

    def toSparse(self, text):
        indices = []
        values = []
        shape = [len(text),0]

        for(batchElement, text) in enumerate(text):

            labelStr = [str.charList.index(c) for c in text]

            if len(labelStr) > shape[1]:
                shape[1]= len(labelStr)

            for(i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices,values,shape)

    def trainBatch(self, batch):

        numBatchOfElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        #Decay the learning rate
        rate = .01 if self.numTrained < 10 else (.001 if self.numTrained< 10000 else .00010)
        evalList = [self.optimizer,]