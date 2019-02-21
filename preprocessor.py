import cv2

path = "/home/matt/Documents/PythonWorkspace/MachineLearning/School/TextRecognition/Data/words/a02/a02-004/a02-004-00-03.png"

def resizeImg(path):

    image = cv2.imread(path)
    h = image.shape[0]
    w = image.shape[1]

    height = 32

    if w == None and h == None:
        return image
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
    cv2.imshow("original", image)
    cv2.imshow("resized",resized)
    print(resized.shape)
    cv2.waitKey(0)



resizeImg(path)