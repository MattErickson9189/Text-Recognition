import tensorflow as tf
import numpy as np
import os


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class NeuralNet:
    #This is a class that will have methods for setting up the CNN and RNN

    size = (128,32)
    maxTextLength = 32
    batchSize = 50
    def __init__(self, learningRate, trainSessions, wordList,decoderType=DecoderType.BestPath, mustRestore = False):

        self.learningRate = learningRate
        self.trainSession = trainSessions
        self.list = wordList
        self.restore = mustRestore
        self.decoder = decoderType
        self.snapID = 0

        self.isTrain = tf.placeholder(tf.bool, name="isTrain")

        self.inputImgs = tf.placeholder(tf.float32, shape=(None, NeuralNet.size[0], NeuralNet.size[1]))

        self.CNN(self)

        self.numTrained = 0
        self.learningRate = tf.placeholder(tf.float, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer= tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        (self.sess, self.saver) = self.setUpTF()



    def setUpTF(self):

        sess=tf.Session()

        saver = tf.train.Saver(max_to_keep=1)
        saveLocation = "./SavedModel"
        latest = tf.train.latest_checkpoint(saveLocation)

        #Checks to see if there is a saved model
        if self.restore and not latest:
            raise Exception('No saved model in: ', saveLocation)

        #Load Model
        if latest:
            print('Loading variables from ', latest)
        else:
            print('Starting with fresh model')
            sess.run(tf.global_variables_initializer())
        return (sess,saver)

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
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[NeuralNet.maxTextLen, None, len(self.charList) + 1])
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
            wordChars = open('./Data/Lists/wordCharList.txt').read().splitlines()[0]
            corpus = open('./Data/Lists/corpus.txt').read()

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
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse, self.seqLen : [NeuralNet.maxTextLength] * numBatchOfElements, self.learningRate : rate, self.isTrain: True}
        (_, lossVal) = self.sess.run(evalList.feedDict)
        self.numTrained += 1
        return lossVal

    def decoderOutputText(self, ctcOut, batchSize):

        encodedLabelStrs = [[] for i in range(batchSize)]

        if self.decoder == DecoderType.WordBeamSearch:
            blank = len(self.list)
            for b in range(batchSize):
                for label in ctcOut[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        else:
            decoded = ctcOut[0][0]

            idxDict = {b : [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]
                encodedLabelStrs[batchElement].append(label)

        return [str().join([self.list[c] for c in labelStr ]) for labelStr in encodedLabelStrs]


    def inferBatch(self, batch, calcProbaility=False, ProbabilityOfGT = False):

        numOfElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbaility else [])
        feedDict = {self.inputImgs: batch.imgs, self.seqLen : [NeuralNet.maxTextLength] * numOfElements, self .isTrain: False}
        evalRes= self.sess.run([self.decoder,self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputText(decoded, numOfElements)

        probs = None
        if calcProbaility:

            sparse= self.toSparse(batch.gtTexts) if ProbabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts: sparse, self.seqLen: [NeuralNet.maxTextLength] * numOfElements, self.isTrain: False}
            lossVals = self.sess.run(evalList,feedDict)
            probs = np.exp(-lossVals)
        return (texts, probs)


    def save(self):
        self.snapID += 1
        self.saver.save(self.sess, './Data/SavedModel/modelSnapshot', global_step=self.snapID)
