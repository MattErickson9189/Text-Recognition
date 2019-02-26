import cv2

def resizeImg(path):

    print(path)
    image = cv2.imread(path)
    print()
    h = image.shape[0]
    w = image.shape[1]

    height = 32

    if w == None and h == None:
        return image
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
    #return resized
    cv2.imshow("img", resized)
    cv2.waitKey(0)

