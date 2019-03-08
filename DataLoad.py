import os
import shutil

path = "./Data/Resized/"
trainPath = "./Data/Resized/training/"
testPath = "./Data/Resized/test/"
wordsList = "./Data/words.txt"


#These next few lines are only to be ran once, it puts the first 70% of the images into a training folder
# trainCout = 0
# testCount = 0
# for (dirpath, dirnames, filenames) in os.walk(path):
#     for files in filenames:
#         if(trainCout < 80500):
#             src = path + files
#             dest = trainPath + files
#             shutil.move(src,dest)
#             trainCout += 1
#             print(dest)
#         else:
#             src = path + files
#             dest = testPath + files
#             shutil.move(src,dest)
#             testCount +=1
#             print(dest)

class DataLoader:

    def __init__(self, filePath, batchSize, textLength):

        self.path = filePath
        self.batchSize = batchSize
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


            #formats the file path to match the words.txt
            fileNameSplit = lineSplit[0].split('-')
            fileName = path + fileNameSplit[0]+ '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            gtText = self.truncateLabel(' '.join(lineSplit[8:]), textLength)
            chars = chars.union(set(list(gtText)))

            self.samples.append(Sample)

