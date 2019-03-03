import os
import shutil

path = "./Data/Resized/"
trainPath = "./Data/Resized/training/"
testPath = "./Data/Resized/test/"
trainCout = 0
testCount = 0
#These next few lines are only to be ran once, it puts the first 70% of the images into a training folder

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