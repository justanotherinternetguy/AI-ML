from dataset import train_data, test_data
from rnn import RNN

# pre-processing of data

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# assign index to each word
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
print(word_to_idx['good'])
print(idx_to_word[0])

# each xi is a vector (one-hot vector - all contain 0s except for a single 1)
# 1 will be at the word's corresponding integer index
# we have 18 unique words = xi is a 18-dimensional one-hot

import numpy as np

def createInputs(text):
  """
  return an array of one-hot vectors representing the words in the input text string
  - text in a string
  - each one-hot has shape (vocab_size, 1)
  """
  
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

def softmax(xs):
  # softmax func to input arr
  return np.exp(xs) / sum(np.exp(xs))

rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs) # [[0.49999588] [0.50000412]]