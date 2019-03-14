import cv2
import os
import numpy as np
def resizeImg(path):

    try:
        print(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h = image.shape[0]
        w = image.shape[1]

        if w is None and h is None:
            print("Damaged Image")
        else:

            desiredHeight = 32
            desiredWidth = 128

            fx = w / desiredWidth
            fy = h / desiredHeight

            f = max(fx, fy)
            size = (max(min(desiredWidth, int(w / f)), 1), max(min(desiredHeight, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)

            # creates the resized image
            resized = cv2.resize(image, size, )

            target = np.ones([desiredHeight,desiredWidth]) * 255
            target[0:size[1], 0:size[0]] = resized

            img = cv2.transpose(target)


            #normalize
            (m,s) = cv2.meanStdDev(img)

            m = m[0][0]
            s = s[0][0]

            img = img - m
            img = img/s if s>0 else img

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

            cv2.imwrite(newLocation, img)

    except AttributeError:
        print("Image couldnt be loaded")