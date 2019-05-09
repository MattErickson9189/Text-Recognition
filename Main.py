from DataLoad import DataLoader, Batch
from NeuralNets import NeuralNet, DecoderType
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
            loss = NeuralNet.train(batch)
            print('Batch: ',index[0],'/',index[1],' Loss: ',loss)
        errorRate = validate(NeuralNet, dataLoader)

        if errorRate < bestCharErrorRate:
            print('Character  error rate improved, saving model')
            bestCharErrorRate= errorRate
            cyclesSinceImprovement = 0
            #NeuralNet.storeModelValues()
            NeuralNet.saver.save(NeuralNet.sess, './Data/SavedModel/SavedModel.txt', global_step=NeuralNet.savedModelId)
            open('./Data/Lists/accuracy.txt', 'a').write('Validation character error rate of saved model: %f%%' % (errorRate * 100.00))

        else:

            print('Char Error Rate Not Improved')
            cyclesSinceImprovement += 1

        if cyclesSinceImprovement >= earlyStopping:
            print('Max cycles since improvement reached...Stopping...')
            break


def validate(NeuralNet, dataLoad):

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
            print('OK' if dist==0 else '[ERROR]', '"'+ batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')


    #Print the results
    charErrorRate = numCharErr /charTotal
    wordAccuracy = numWordOK / wordTotal
    print('Character Error rate: %f%%. Word Accuracy: %f%%.' % (charErrorRate*100.00, wordAccuracy * 100.00))
    return charErrorRate



def infer(model, fnImg):

    batch= Batch(None, fnImg)
    (recognized, probability) = NeuralNet.inferBatch(batch, True)

    print('Recognized: ', '"' + recognized[0] + '"')
    print('Probability: ', '"' + probability[0])


def Main():

    #option = input('What action would you like to take?')
    option = "train"

    #If train or validate
    if option == 'train' or option == 'validate':

        loader = DataLoader(NeuralNet.batchSize, NeuralNet.maxTextLen, NeuralNet.imgSize)

        #save characters for reference
        open('./Data/Lists/charList.txt', 'w').write(str().join(loader.charList))

        #Save words for reference
        open('./Data/Lists/corpus.txt', 'w').write(str().join(loader.trainingWords + loader.testWords))

        if option == 'train':
            model = NeuralNet(loader.charList, DecoderType.BestPath )
            train(model,loader)
        else:
            model = NeuralNet(loader.charList, DecoderType.BestPath, mustRestore=True)
            validate(model,loader)

Main()