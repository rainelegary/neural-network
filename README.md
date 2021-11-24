# neural-network

This application uses the reinforcement learning model of machine learning to try approximating a function of x.

Please run the file main.py to run the program, then enter the size of the neural network in the console. After this, watch the function evolve in the console! The larger the network, the longer it will take to calculate, so it's recommended to use a small network such as 2 hidden layers, each with 6 nodes.

It starts out by guessing an arbitrary function, then calculates its margin of error from the function it's aiming for by using the method of least squares.
Based on this error, it then calculates its learning rate. This learning rate determines how much the weights and biases in the neural network can change in its next mutation.
The smaller the error, the slower the learning rate which allows the network to converge on a local minimum of error.
