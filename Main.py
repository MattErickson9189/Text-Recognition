from DataLoad import DataLoader, Batch
from NeuralNets import NeuralNet



def train(NeuralNet, dataLoader):

    epoch = 0

    bestCharErrorRate = float('inf')
    #Keeps track of how long since the model improved
    cyclesSinceImprovement = 0

    #If there is no improvement for 5 epochs stop
    earlyStopping = 5

    while True:

        epoch += 1
        print('Epoch: ', epoch)

        #This is when the training starts
        print('Starting Training')
        dataLoader.trainSet()
        while dataLoader.hasNext():
            index = dataLoader.getIndex()
            batch = dataLoader.getNext()
            loss = NeuralNet