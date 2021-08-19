import math

class TrainingDataSet:
    def __init__(self, dataSize):
        self.dataSize = dataSize
        self.trainingData = {}
        self.generateTrainingData()


    def generateTrainingData(self):
        dataSize = self.dataSize
        inputs = []
        outputs = []
        for i in range(dataSize):
            x = 2*i/dataSize - 1
            y = self.outputGoal(x)

            inputs.append([x])
            outputs.append([y])

        self.trainingData['in'] = inputs
        self.trainingData['out'] = outputs


class Oscillation(TrainingDataSet):
    def __init__(self, dataSize, amplitude, frequency, xOff, yOff):
        self.amplitude = amplitude
        self.frequency = frequency
        self.xOff = xOff
        self.yOff = yOff

        super().__init__(dataSize)

    def outputGoal(self, x):
        y = math.sin(self.frequency*(x-self.xOff)) * self.amplitude + self.yOff
        y = math.sin(4*x) / 4
        y = (pow(math.e, x) - 2)/(-1.5)
        return y



