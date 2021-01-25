import numpy as np
import scipy.special

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

    def train():
        pass
    
    def query(self, inputsList):
        inputs = np.array(inputsList, ndmin=2).T
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs

inputNodes = 3
hiddenNodes = 3
outputNodes = 3

learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
print(n.query([1, 0.5, -1.5]))