from random import *
from training import *
import math
import sys
import copy


class NetworkMonitor:
    pass


class Network:
    def __init__(self, initializer, trainingData):
        self.trainingData = trainingData

        setArgs = copy.deepcopy(initializer.allArgs)

        self.hLayerCount = initializer.allArgs['hidden layer count']
        self.layerSize = initializer.allArgs['layer size']
        self.populations = initializer.allArgs['populations']
        self.layerCount = self.hLayerCount + 2

        self.layerSizes = [1]
        for i in range(self.hLayerCount):
            self.layerSizes.append(self.layerSize)
        self.layerSizes.append(1)


        self.connectionsDict = self.generateConnections()
        self.learningRate = 1
        self.maxBias = 1  # adjustable
        self.maxWeight = 2  # adjustable

        self.currentError = self.calcError(self.connectionsDict, self.trainingData)
        self.bestNets = {self.currentError: self.connectionsDict}
        self.childNets = {}

    def generateConnections(self):
        connectionsDict = {'weights': {}, 'biases': {}}

        lSizes = self.layerSizes
        layerCount = self.layerCount
        for layerA in range(layerCount - 1):
            nameLayerA = 'layerA' + str(layerA)
            connectionsDict['weights'][nameLayerA] = {}

            for nodeA in range(lSizes[layerA]):
                nameNodeA = 'nodeA' + str(nodeA)
                connectionsDict['weights'][nameLayerA][nameNodeA] = {}
                for nodeB in range(lSizes[layerA + 1]):
                    nameNodeB = 'nodeB' + str(nodeB)
                    connectionsDict['weights'][nameLayerA][nameNodeA][nameNodeB] = uniform(-1, 1)

        for layerB in range(1, layerCount):
            nameLayerB = 'layer' + str(layerB)
            connectionsDict['biases'][nameLayerB] = {}
            for nodeB in range(lSizes[layerB]):
                nameNodeB = 'node' + str(nodeB)
                bias = uniform(-1, 1)
                connectionsDict['biases'][nameLayerB][nameNodeB] = bias
                
        return connectionsDict


    def runNetworkInput(self, connections, inputValues=()):
        lSizes = self.layerSizes
        layerCount = self.layerCount
        weights = connections['weights']
        biases = connections['biases']
        nextLayerNodes = inputValues
        for layerA in range(layerCount - 1):
            layerB = layerA + 1
            currentLayerNodes = nextLayerNodes
            nextLayerNodes = []
            for nodeB in range(lSizes[layerB]):
                genVal = 0
                bias = biases['layer' + str(layerB)]['node' + str(nodeB)]
                for nodeA in range(lSizes[layerA]):
                    usedWeight = weights['layerA' + str(layerA)]['nodeA' + str(nodeA)]['nodeB' + str(nodeB)]
                    genVal += currentLayerNodes[nodeA] * usedWeight

                genVal += bias
                nodeB_value = sigmoid(genVal)
                nextLayerNodes.append(nodeB_value)

        outputNodes = nextLayerNodes
        return outputNodes


    def selectBestNets(self, trainingData, numNets):

        childNets = copy.deepcopy(self.childNets)
        bestNets = {}
        leastError = math.inf

        errorCutoff = math.inf
        for childNet in childNets:
            connections = childNets[childNet]
            error = self.calcError(connections, trainingData)
            if error < errorCutoff:
                if len(bestNets) >= numNets:
                    bestNets.pop(errorCutoff)
                bestNets[error] = connections

                mostError = 0
                for net in bestNets:
                    if net > mostError:
                        mostError = net
                    if net < leastError:
                        leastError = net

                errorCutoff = mostError


        self.connectionsDict = copy.deepcopy(bestNets[leastError])
        self.bestNets = copy.deepcopy(bestNets)


    def calcError(self, connections, trainingData):
        inputs = trainingData['in']
        outputs = trainingData['out']
        dataSize = len(inputs)

        squaredDiffs = []
        for inputN in range(dataSize):
            tOutput = outputs[inputN][0]
            nOutput = self.runNetworkInput(connections, inputs[inputN])[0]
            squaredDiff = pow(nOutput - tOutput, 2)
            squaredDiffs.append(squaredDiff)

        totalSqDiff = 0
        for squaredDiff in squaredDiffs:
            totalSqDiff += squaredDiff

        error = totalSqDiff/dataSize
        return error


    def updateLearningRate(self, error):
        x = error
        l = 0.1
        l = 1
        l = pow(x, 1.35) / 5
        l = x
        self.learningRate = l


    def mutateConnections(self):
        trainingData = self.trainingData
        pops = self.populations
        mutated, new, duplicated = pops['mutated'], pops['new'], pops['duplicated']

        if self.childNets:
            numNetsChosen = 3   # adjustable
            self.selectBestNets(trainingData, numNetsChosen)
        bestNets = copy.deepcopy(self.bestNets)

        childNets = {}
        leastError = math.inf
        nBest = 0
        for best in bestNets:
            if best < leastError:
                leastError = best
            parentConnections = copy.deepcopy(bestNets[best])
            for mutationN in range(mutated + new + duplicated):
                mutationMode = 'mutated'
                if mutationN >= mutated:
                    mutationMode = 'new'
                if mutationN >= mutated + new:
                    mutationMode = 'duplicated'


                connections = copy.deepcopy(self.mutationProcess(parentConnections, mutationMode))
                childNets['child%d-%d' % (nBest, mutationN)] = copy.deepcopy(connections)

            nBest += 1

        self.childNets = childNets
        self.currentError = self.calcError(self.connectionsDict, self.trainingData)
        self.updateLearningRate(self.currentError)
        print('error: ' + str(self.currentError))
        print('learning rate: ' + str(self.learningRate))



    def mutationProcess(self, connections, mutationMethod):
        maxBias = self.maxBias
        maxWeight = self.maxWeight
        learningRate = self.learningRate
        conns = copy.deepcopy(connections)

        weights = copy.deepcopy(conns['weights'])
        biases = copy.deepcopy(conns['biases'])

        for layerA in weights:
            for nodeA in weights[layerA]:
                for nodeB in weights[layerA][nodeA]:
                    weight = 0

                    if mutationMethod == 'mutated':
                        weight = weights[layerA][nodeA][nodeB]
                        weight += uniform(-maxWeight, maxWeight) * learningRate
                        if weight > maxWeight:
                            weight = maxWeight
                        if weight < -maxWeight:
                            weight = -maxWeight

                    if mutationMethod == 'new':
                        weight = uniform(-maxWeight, maxWeight)

                    if mutationMethod == 'duplicated':
                        weight = weights[layerA][nodeA][nodeB]

                    conns['weights'][layerA][nodeA][nodeB] = weight

        for layer in biases:
            for node in biases[layer]:
                bias = 0

                if mutationMethod == 'mutated':
                    bias = biases[layer][node]
                    bias += uniform(-maxBias, maxBias) * learningRate
                    if bias > maxBias:
                        bias = maxBias
                    if bias < -maxBias:
                        bias = -maxBias

                if mutationMethod == 'new':
                    bias = uniform(-maxBias, maxBias)

                if mutationMethod == 'duplicated':
                    bias = biases[layer][node]


                conns['biases'][layer][node] = bias
        return conns


class NetworkInitializer:
    def __init__(self):
        self.allArgs = {}
        inputMode = 'user input'

        if inputMode == 'user input':
            self.getNetworkArgsUserInput()


    def getNetworkArgsUserInput(self):
        print("""
Please enter the following information...
type \"exit\" at any point to terminate this process. """)
        questionDict = {
            'hidden layer count': 'How many hidden layers in the network? ',
            'layer size': 'How many nodes in each layer? ',
            #'layer sizes': None
                        }

        intAttributes = ['hidden layer count', 'layer size']
        attrValues = {}

        for attribute in questionDict:
            if attribute != 'layer sizes':
                questionString = questionDict[attribute]
                attributeInput = input(questionString)

                if attributeInput.lower() == 'exit':
                    sys.exit("exiting program")

                if attribute in intAttributes:
                    attrValues[attribute] = int(attributeInput)
                else:
                    attrValues[attribute] = attributeInput

            if attribute == 'layer sizes':
                layerSizes = []
                for layerN in range(attrValues['hidden layer count'] + 2):

                    layerName = 'hidden layer %d' % layerN
                    if layerN == 0:
                        layerName = 'the input layer'
                    if layerN == attrValues['hidden layer count'] + 1:
                        layerName = 'the output layer'

                    questionString = 'how many nodes in %s? ' % layerName

                    attributeInput = input(questionString)

                    if attributeInput.lower() == 'exit':
                        sys.exit("exiting program")

                    layerSize = int(attributeInput)
                    layerSizes.append(layerSize)
                attrValues['layer sizes'] = layerSizes

        attrValues['populations'] = {'mutated': 10, 'new': 0, 'duplicated': 1}  # adjustable
        self.allArgs = attrValues



def sigmoid(x):
    return (1-pow(math.e, -2*x))/(1+pow(math.e, -2*x))  # adjustable
