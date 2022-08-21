# COMBINING NEURONS TO CREATE A NETWORK!

import numpy as np

def sigmoid(x):
    # Function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights;
        self.bias = bias;

    def feedforward(self, inputs):
        dotted_total_val = np.dot(self.weights, inputs) + self.bias
        return sigmoid(dotted_total_val)

weights = np.array([0, 1])
bias = 4

n1 = Neuron(weights, bias)

x = np.array([2, 3])
print("Single neuron: " , n1.feedforward(x)) # 0.9990889488055994

# Create a hidden layer as well as + neurons
# x1, x2 (input) --> h1, h2 (hidden) --> o1 (output)
# Using the same numerical example from `neuron.py`,

"""
Let h1, h2, o1 denote the OUTPUTS of the neurons that they represent
Assume that all neurons have w = [0, 1] and b = 0


Passing input x = [2, 3]

HIDDEN LAYER:
h1 = h2 = f(w * x + b)
= f((0 * 2) + (1 * 3) + 0)
= f(3)
= 0.9526

OUTPUT LAYER
o1 = f(w * [h1, h2] + b)
= f((0 * h1) + (1 * h2) + 0)
= f(0.9526)
= 0.7216
"""

class NeuralNetwork:
    """
    - 2 inputs
    - input: [x1]
    - hidden: [h1, h2]
    - output: [o1]

    - constant w = [0, 1]
    - constant b = 0
    """
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = NeuralNetwork();
# x is already defined above
print("Neural network: " , network.feedforward(x)) # 0.7216325609518421

