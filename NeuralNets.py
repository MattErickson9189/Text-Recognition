import tensorflow as tf
import numpy as np
import sys

class decodeType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class NeuralNet:
    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decodeType=decodeType.BestPath, mustRestore=False):

        self.charList = charList
        self.decodeType = decodeType
        self.mustRestore = mustRestore
        self.savedModelId = 0

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
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.calculatedLoss)

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

    # TODO Refactor
    def RNN(self):

        recurrentInput = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        hiddenValues = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=hiddenValues, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        ((forward, backward), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=recurrentInput,
                                                        dtype=recurrentInput.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        # project output to chars
        kernel = tf.Variable(tf.truncated_normal([1, 1, hiddenValues * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOutput = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


    def CTC(self):

        self.ctcInput = tf.transpose(self.rnnOutput, [1, 0, 2])
        self.truthTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]),
                                       tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.sequenceLength = tf.placeholder(tf.int32, [None])
        self.calculatedLoss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.truthTexts, inputs=self.ctcInput, sequence_length=self.sequenceLength,
                           ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32, shape=[NeuralNet.maxTextLen, None, len(self.charList) + 1])
        self.calculatedLossPerElement = tf.nn.ctc_loss(labels=self.truthTexts, inputs=self.savedCtcInput,
                                             sequence_length=self.sequenceLength, ctc_merge_repeated=True)

        if self.decodeType == decodeType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcInput, sequence_length=self.sequenceLength)
        elif self.decodeType == decodeType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcInput, sequence_length=self.sequenceLength,
                                                         beam_width=50, merge_repeated=False)
        elif self.decodeType == decodeType.WordBeamSearch:
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information
            chars = str().join(self.charList)
            wordChars = open('./Data/Lists/wordCharList.txt').read().splitlines()[0]
            completedWords = open('../Data/Lists/completedWords.txt').read()

            # decode using the "Words" mode of word beam search
            self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcInput, dim=2), 50, 'Words', 0.0, completedWords.encode('utf8'), chars.encode('utf8'),wordChars.encode('utf8'))

    def setupTF(self):

        sess = tf.Session()  # TF session

        saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = './Data/SavedModel/'

        currentModel = tf.train.latest_checkpoint(modelDir)


        if self.mustRestore and not currentModel:
            print('No saved model found in: ' + modelDir)
            exit(1)

        # load saved model
        if currentModel:
            print('Loading values from ' + currentModel)
            saver.restore(sess, currentModel)
        else:
            print('Loaded with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)


    def train(self, batch):

        batchElements = len(batch.imgs)
        sparse = self.tensorSparse(batch.truthTexts)
        #Learning rate
        rate = 0.01
        list = [self.optimizer, self.calculatedLoss]
        dictionary = {self.inputImgs: batch.imgs, self.truthTexts: sparse,self.sequenceLength: [NeuralNet.maxTextLen] * batchElements, self.learningRate: rate, self.is_train: True}
        (_, totalLoss) = self.sess.run(list, dictionary)
        self.totalBatches += 1
        return totalLoss

    def tensorSparse(self, labels):

        indices = []
        values = []
        shape = [len(labels), 0]

        # go over all texts
        for (batchElement, text) in enumerate(labels):
            # convert to string of label (i.e. class-ids)
           stringLabel = [self.charList.index(c) for c in text]

           # Checking the size of the tensor
           tensorSize = len(stringLabel)
           if tensorSize > shape[1]:
              shape[1] = len(stringLabel)
            # put each label into sparse tensor
           for (i, label) in enumerate(stringLabel):
              indices.append([batchElement, i])
              values.append(label)

        return (indices, values, shape)

    def decodeToText(self, ctcOutput, batchSize):


        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decodeType == decodeType.WordBeamSearch:
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

    # TODO Refactor
    def predictOnBatch(self, batch, calcProbability=False, probabilityOfGT=False):

        # decode each of the labels
        batchElements = len(batch.imgs)
        dictionary = {self.inputImgs: batch.imgs, self.sequenceLength: [NeuralNet.maxTextLen] * batchElements,
                      self.is_train: False}
        evaluationRes = self.sess.run([self.decoder, self.ctcInput], dictionary)
        decoded = evaluationRes[0]
        wordText = self.decodeToText(decoded, batchElements)

        # load recognized text into the RNN
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.truthTexts) if probabilityOfGT else self.toSparse(wordText)
            ctcInput = evaluationRes[1]
            list = self.calculatedLossPerElement
            dictionary = {self.savedCtcInput: ctcInput, self.truthTexts: sparse,
                          self.sequenceLength: [NeuralNet.maxTextLen] * batchElements, self.is_train: False}
            lossVals = self.sess.run(list, dictionary)
            probs = np.exp(-lossVals)
        return (wordText, probs)
