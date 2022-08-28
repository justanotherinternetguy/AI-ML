import numpy as np
from numpy.random import randn

class RNN:
  def __init__(self, input_size, output_size, hidden_size=64):
    # weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000
    
    # biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))
    
  # RNN's forward pass with tanh
  def forward(self, inputs):
    """
    Returns the final output and the hidden state
    - inputs: array of one-hot vectors as shape: (input_size, 1)
    """
    h = np.zeros((self.Whh.shape[0], 1))
    
    # perform each step of RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      
      # output
      y = self.Why @ h + self.by
      
      return y, h