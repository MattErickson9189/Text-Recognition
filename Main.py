from DataLoad import DataLoader, Batch
from NeuralNets import NeuralNet
import editdistance

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

        if errorRate < bestCharErrorRate:
            print('Character  error rate improved, saving model')
            bestCharErrorRate= errorRate
            cyclesSinceImprovement = 0
            NeuralNet.save()
            open('./Data/Lists/accuracy.txt', 'a').write('Validation character error rate of saved model: %f%%' % (errorRate * 100.00))

        else:

            print('Char Error Rate Not Improved')
            cyclesSinceImprovement += 1

        if cyclesSinceImprovement >= earlyStopping:
            print('Max cycles since improvement reached...Stopping...')
            break


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
        (recognized,_) = NeuralNet.inferBatch(batch)

        print('Ground Truth -> Recognized')
        for i in range(len(recognized)):

            numWordOK +=1 if batch.gtTexts[i] == recognized[i] else 0
            wordTotal +=1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr +=dist
            charTotal += len(batch.gtTexts[i])
            print('OK' if dist==0 else '[ERROR %d]' %dist, '"'+ batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')


    #Print the results
    charErrorRate = numCharErr /charTotal
    wordAccuracy = numWordOK / wordTotal
    print('Character Error rate: %f%%. Word Accuracy: %f%%.' % (charErrorRate*100.00, wordAccuracy * 100.00))
    return charErrorRate