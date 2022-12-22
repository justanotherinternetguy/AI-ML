# CLEAR GOAL: Minimize the loss of the neural network
# We know that we can change the NN's weights and biases to influence its preds, but how can we do that in a way that minimizes loss?

# multivariable calculus moment...

# We can wrote loss as a multivar function --> L(w1, w2, w3, w4, w5, w6, b1, b2, b3)

# Imagine we want to modify w1. How would L() change if we changed w1?
# Use partial derivatives

# After some math... We can break a partial derivative down into serveral easier-to-calculate parts
# This system of calculating partial derivatives by working backwards is called BACKPROPOGATION

# Stochastic Gradient Descent
# tells us how to change our weights and biases to minimize loss.
# Equation w1 = w1 - n * (partial deriv)
# If (deriv) is >0, w1 will decrease, which decreases L
# If (deriv) is <0, w1 will increase, which decreases L
# If we do this for every weight and bias in the network, the loss will slowly decrease and our network will improve.


""" Training Process
  Choose one sample from our dataset. This is what makes it stochastic gradient descent - we only operate on one sample at a time.
  Calculate all the partial derivatives of loss with respect to weights or biases.
  Use the update equation to update each weight and bias.
  Go back to step 1.
"""
