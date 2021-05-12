import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activationFunction = lambda x:scipy.special.expit(x)
        pass

    def train(self, inputsList, targetsList):
        inputs = np.array(inputsList, ndmin=2).T
        targets = np.array(targetsList, ndmin=2).T

        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        outputErrors = targets-finalOutputs
        hiddenErrors = np.dot(self.who.T, outputErrors)

        self.who += self.lr*np.dot((outputErrors*finalOutputs*(1-finalOutputs)), np.transpose(hiddenOutputs))
        self.wih += self.lr*np.dot((hiddenErrors*hiddenOutputs*(1-hiddenOutputs)), np.transpose(inputs))
        pass
    
    def query(self, inputsList):
        inputs = np.array(inputsList, ndmin=2).T

        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs

inputNodes = 784
hiddenNodes = 100
outputNodes = 10

learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

trainingDataFile = open("MNIST_Data/mnist_train_100.csv", 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

for record in trainingDataList:
    allValues = record.split(',')
    inputs = (np.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(outputNodes)+0.01
    targets[int(allValues[0])] = 0.99
    n.train(inputs, targets)
    pass

testDataFile = open("MNIST_Data/mnist_test_10.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

allValues = testDataList[0].split(',')
iamgeArray = np.asfarray(allValues[1:]).reshape((28,28))
plt.imshow(iamgeArray, cmap="Greys", interpolation='None')
plt.show()

print(allValues[0])