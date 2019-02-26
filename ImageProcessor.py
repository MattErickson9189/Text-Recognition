import os, sys
from preprocessor import resizeImg
import cv2
path = "./Data/Test/"

newLocation = path + "resizedImages"

if(os.path.isdir(newLocation == False)):
    os.mkdir(newLocation)



for(dirpath, dirnames, filenames) in os.walk(path):
    for files in filenames:
        relative = os.path.join(dirpath,files)
        print(relative)
        Image = resizeImg(relative)
        cv2.imshow("Img",Image)
        cv2.waitKey(0)
