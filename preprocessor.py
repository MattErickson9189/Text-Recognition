import cv2
import os

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