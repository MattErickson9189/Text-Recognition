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
            loss = NeuralNet.trainBatch(batch)
            print('Batch: ',index[0],'/',index[1],' Loss: ',loss)

            errorRate = validate(NeuralNet, dataLoader)



def validate(model, dataLoad):

    print('Validating')
    dataLoad.trainSet()
    numCharErr = 0
    charTotal = 0
    numWordOK = 0
    wordTotal = 0

    while(dataLoad.hasNext()):

        index = dataLoad.getIndex()
        print('Batch:', index[0],'/', index[1])
        batch = dataLoad.getNext()
        (recognized,_) = NeuralNet