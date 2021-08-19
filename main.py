from networks import *
from graphics import *
import time

def main():
    myWindow = WindowSet('the window', '800x600')

    sinWave = Oscillation(100, 1, math.pi, 0, 0)
    sinWaveTraining = sinWave.trainingData

    testInit = NetworkInitializer()
    testNet = Network(testInit, sinWaveTraining)

    while True:
        testNet.mutateConnections()
        time.sleep(10)

        points = []
        for i in range(100):
            x = i/50 - 1
            out = testNet.runNetworkInput(testNet.connectionsDict, [x])[0]
            expectedOut = sinWave.outputGoal(x)
            points.append(coordsToPixel([x, out]))
            points.append(coordsToPixel([x, expectedOut]))

        loopWindow(myWindow, points=points)


main()

