import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

import pylab as pl

pl.gray()
pl.matshow(digits.images[0])


images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
    plt.show()

