# We will attempt to expand on our NN by applying a real-world example to "train" the NN

# Goal: Predict someone's gender given their weight and height

# [x1: weight, x2: height] --> [h1, h2] --> [o1: gender]

# data/train.csv : raw data
# data/train_cleaned.csv: data shifted

# USE MSE (Mean squared error) loss function
# When you train a network, you want to minimize its losses


# MSE
import numpy as np

def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5