import os
from preprocessor import resizeImg
import cv2
path = "./Data/words/"
count = 0

for(dirpath, dirnames, filenames) in os.walk(path):
    for files in filenames:
        count= count+1
        print("Files Resized:", count)
        print(files)
        relative = os.path.join(dirpath,files)
        resizeImg(relative)


#I wrote the code below to check to see how many files were successful
# Only two were skipped and one I did manually
# path2 = "./Data/Resized/"
# count2 =0
# for(dirpath,dirnames, filenames) in os.walk(path):
#     for files in filenames:
#         count = count +1
#
# for(dirpath,dirnames,filenames) in os.walk(path2):
#     for files in filenames:
#         count2 = count2 + 1
#
#
# print("Images in words directory:", count)
# print("Images in Resized directory:", count2)