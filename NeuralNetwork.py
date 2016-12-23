import mnist_loader
import numpy as np

class NeuralNetwork:

    def __init__(self, trainingDataCount, testDataCount, hiddenLayerCount, hlNodeCount, batchSize, learningRate ,maxIteration):

        print("--Reading Data")
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

        self.inputs = training_data[0][0]
        for i in range(trainingDataCount - 1):
            self.inputs = np.column_stack([self.inputs, training_data[i + 1][0]])

        self.targets = training_data[0][1]
        for i in range(trainingDataCount - 1):
            self.targets = np.column_stack([self.targets, training_data[i + 1][1]])

        self.tests = test_data[0][0]
        for i in range(testDataCount - 1):
            self.tests = np.column_stack([self.tests, test_data[i + 1][0]])

        self.testTargets = []
        for i in range(testDataCount):
            self.testTargets.append(test_data[i][1])

        self.hiddenLayerCount = hiddenLayerCount
        self.hlNodeCount = hlNodeCount
        self.trainingDataCount = trainingDataCount
        self.testDataCount = testDataCount
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.maxIteration = maxIteration

        self.initialize_weights()

    def initialize_weights(self):
        print("--Initializing Weights")
        self.weights = []
        self.biasWeights = []

        if (self.hiddenLayerCount == 0):
            self.weights.append(np.random.randn(10, 784) )
            self.biasWeights.append(np.random.randn(10, 1) )
        else:
            self.weights.append(np.random.randn(self.hlNodeCount, 784) )
            self.biasWeights.append(np.random.randn(self.hlNodeCount, 1) )
            for i in range(self.hiddenLayerCount - 1):
                self.weights.append(np.random.randn(self.hlNodeCount, self.hlNodeCount) )
                self.biasWeights.append(np.random.randn(self.hlNodeCount, 1) )
            self.weights.append(np.random.randn(10, self.hlNodeCount) )
            self.biasWeights.append(np.random.randn(10, 1) )

        print("--Weights are Initialized:")
        print("----Dimensions of Input and Hidden Layer's Weights :", end="")
        for i in range(len(self.weights)):
            print(self.weights[i].shape, end="")

        print()

        print("----Dimensions of Bias Weights:", end="")
        for i in range(len(self.weights)):
            print(self.biasWeights[i].shape, end="")
        print()

    def train(self):
        print("--Starting Train:")
        for l in range(self.maxIteration):
            error = 0

            for batch_index in range(self.trainingDataCount // self.batchSize):

                # Forward-propagation

                nets = []  # before activation function
                outs = []  # after activation function

                inputs_batch    = self.inputs [:, batch_index * self.batchSize : (batch_index + 1) * self.batchSize]
                targets_batch   = self.targets[:, batch_index * self.batchSize : (batch_index + 1) * self.batchSize]

                outs.append(inputs_batch)

                for j in range(self.hiddenLayerCount + 1):
                    nets.append(np.dot(self.weights[j], outs[j]) + self.biasWeights[j])
                    outs.append(sigmoid(nets[j]))

                error += np.square(targets_batch - outs[-1])

                # Back-propagation

                deltas = []
                deltas.append(np.multiply(-1 * rowMean((targets_batch - outs[-1])),
                                          np.multiply(rowMean(outs[-1]), (1 - rowMean(outs[-1])))))
                for k in range(self.hiddenLayerCount):
                    deltas.append(np.multiply(np.dot(self.weights[-1 * (k + 1)].T, deltas[k]),
                                              np.multiply(rowMean(outs[-1 * (k + 2)]),
                                                          (1 - rowMean(outs[-1 * (k + 2)])))))

                for k in range(len(self.weights)):
                    #print(np.sum(np.square(deltas[-1 * (k + 1)])))
                    self.weights[k] = self.weights[k] - self.learningRate * np.dot(deltas[-1 * (k + 1)], rowMean(outs[k]).T)
                    self.biasWeights[k] = self.biasWeights[k] - self.learningRate * deltas[-1 * (k + 1)]

            print("----#%d Mean Squared Error:%f"%(l,np.mean(error)))
        print("--Train Finished")

    def test(self):
        print("--Starting Test")

        accuracy = 0

        nets = []  # before activation function
        outs = []  # after activation function

        outs.append(self.tests)

        for j in range(self.hiddenLayerCount + 1):
            nets.append(np.dot(self.weights[j], outs[j]) + self.biasWeights[j])
            outs.append(sigmoid(nets[j]))

        for i in range(self.testDataCount):
            if (np.argmax([outs[-1][:, i]]) == self.testTargets[i]):
                accuracy = accuracy + 1

        print("--Test Finished")
        print("--Accuracy on Test Data: %%%d"%((accuracy/self.testDataCount)*100) )


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

def rowMean(x):
    return np.array([np.mean(x,axis=1)]).T


# ----

# Parameters: TrainDataSize, TestDataSize, HiddenLayerCount, HiddenLayerNodeCount, BatchSize, learningRate, MaxIter
NN = NeuralNetwork(2000,200,2,50,10,0.1,200)  # if hiddenLayerCount = 0, that means it has only inpt and output nodes)
NN.train()                                    # if hiddenLayerCount = x, that means is has x+1 layer.
NN.test()


