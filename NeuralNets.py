import tensorflow as tf
import numpy as np
import os
import sys

class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class NeuralNet:
    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, NeuralNet.imgSize[0], NeuralNet.imgSize[1]))

        # setup CNN, RNN and CTC
        self.CNN()
        self.RNN()
        self.CTC()

        # setup optimizer to train NN
        self.totalBatches = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def CNN(self):

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [7, 7, 5,5, 3, 3, 3]
        #List of the features for the layers
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(
                tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.cnnOut4d = pool

    def RNN(self):
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


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
            chars = str().join(self.charList)
            wordChars = open('./Data/Lists/wordCharList.txt').read().splitlines()[0]
            corpus = open('../Data/Lists/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words',
                                                                    0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                                                    wordChars.encode('utf8'))

    def setupTF(self):
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.Session()  # TF session

        saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = './Data/SavedModel/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):

        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):


        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def train(self, batch):

        batchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        #Learning rate
        rate = 0.01
        list = [self.optimizer, self.loss]
        dictionary = {self.inputImgs: batch.imgs, self.gtTexts: sparse,
                    self.seqLen: [NeuralNet.maxTextLen] * batchElements, self.learningRate: rate, self.is_train: True}
        (_, lossVal) = self.sess.run(list, dictionary)
        self.totalBatches += 1
        return lossVal

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):

        # decode each of the labels
        batchElements = len(batch.imgs)
        list = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        dictionary = {self.inputImgs: batch.imgs, self.seqLen: [NeuralNet.maxTextLen] * batchElements,
                    self.is_train: False}
        evaluationRes = self.sess.run([self.decoder, self.ctcIn3dTBC], dictionary)
        decoded = evaluationRes[0]
        wordText = self.decoderOutputToText(decoded, batchElements)

        # load recognized text into the RNN
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(wordText)
            ctcInput = evaluationRes[1]
            list = self.lossPerElement
            dictionary = {self.savedCtcInput: ctcInput, self.gtTexts: sparse,
                        self.seqLen: [NeuralNet.maxTextLen] * batchElements, self.is_train: False}
            lossVals = self.sess.run(list, dictionary)
            probs = np.exp(-lossVals)
        return (wordText, probs)

    def save(self):

        self.snapID += 1
        self.saver.save(self.sess, './Data/SavedModel/SavedModel.txt', global_step=self.snapID)
