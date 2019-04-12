import random
import numpy as np
import cv2
from preprocessor import individualProcess
path = "./Data/Resized/"
wordsList = "./Data/words.txt"


class data:

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoader:

    def __init__(self, batchSize, textLength, imgSize):

        self.DataAugmentation = False
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.index = 0
        self.textLength = textLength

        #store the images in the batch
        self.images = []


        #Loads the word list
        list = open(wordsList)

        chars = set()

        for line in list:
            if not line or line[0] =='#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            #adds -Resized.png to the file name to match the resized directory
            filePath = path + lineSplit[0] + '-Resized.png'

            #gtText is what the writing in the image is
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), textLength)
            #chars = chars.union(set(list(gtText)))
            chars = chars.union(set(gtText))
            self.images.append(data(gtText,filePath))

        #Split the data and store into lists
        splitData = int(.90 * len(self.images))
        self.training = self.images[:splitData]
        self.testing = self.images[splitData:]

        #get the corresponding text from the words list
        self.trainingWords = [x.gtText for x in self.training]
        self.testWords = [x.gtText for x in self.testing]

        #Number of images chosen for trainingn per epoch
        self.imagesPerEpoch = 25000


        #Start Training
        self.trainSet()

        #List all the chars in the set
        #self.charList = sorted(list(chars))
        self.charList = sorted(chars)

    def trainSet(self):
        self.DataAugmentation = True
        self.index = 0
        random.shuffle(self.training)
        self.images = self.training[:self.imagesPerEpoch]


    def testSet(self):
        self.DataAugmentation = False
        self.index = 0
        self.images = self.testing


    def getIndex(self):
        return (self.index // self.batchSize +1, len(self.images) // self.batchSize)

    def hasNext(self):
        return self.index + self.batchSize <= len(self.images)

    def getNext(self):
        batchRange = range(self.index, self.index + self.batchSize)
        gtTexts = [self.images[i].gtText for i in batchRange]

        imgs = [individualProcess(cv2.imread(self.images[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.DataAugmentation) for i in batchRange]
        self.index += self.batchSize

        return Batch(gtTexts, imgs)

    def truncateLabel(self, text, MaxTextLength):
        #Maps the fileName to the name in the words list
        cost =0
        for i in range(len(text)):
            if i != 0 and text[1] == text[i-1]:
                cost+=2
            else:
                cost+=1
            if cost>MaxTextLength:
                return text[:i]
        return text
