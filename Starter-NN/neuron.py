# Neuron takes inputs, does math w/ inputs, and produces an output

# y = f(x1 * w1 + x2 * w2 + b)
# Input * weight
# + bias constant
# f() activation
  # Common activation function is sigmoid
  # Compression of values between (0, 1)


# ASSUME
# w = [0, 1] (vector form) (aka w1 = 0, w2 = 1)
# b = 4
# activation=sigmoid


# give neuron INPUT x = [2, 3], using dot product

"""
(w * x) + b = ((w1 * x1) + (w2 * x2)) + b
= 0 * 2 + 1 * 3 + 4
= 7

y = f(w * x + b) = f(7) => (sigmoid) => 0.999
"""
# this process is called FEEDFORWARD

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
print(n1.feedforward(x)) # 0.9990889488055994
