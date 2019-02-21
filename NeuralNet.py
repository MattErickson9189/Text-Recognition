import math
import numpy as np

# sends the number through a sigmoid function in order to bring it between the range of 0 and 1
def sigmoid(x):
    return 1 / (1 + math.exp(-5 * x))


def tahn(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


np.random.seed(1)

