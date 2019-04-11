import cv2
import os
import numpy as np
import random
def resizeImg(path):

    try:
        print(path)
        image = cv2.imread(path)
        h = image.shape[0]
        w = image.shape[1]

        height = 32

        if w is None and h is None:
            print("Damaged Image")
        else:
            r = height / float(h)
            dim = (int(w * r), height)

            # creates the resized image
            resized = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)

            # gets the files extension
            ext = os.path.splitext(path)[1]

            # gets the files base name and takes the file extension off
            baseName = os.path.basename(path)
            baseName = os.path.splitext(baseName)[0]

            # Sets the resized name
            newName = baseName + "-Resized" + ext

            newLocation = "./Data/Resized/" + newName
            print(newLocation)
            print()

            cv2.imwrite(newLocation, resized)

    except AttributeError:
        print("Image couldnt be loaded")


def individualProcess(img, imgSize, dataAugmentation = False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

        # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

        # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img