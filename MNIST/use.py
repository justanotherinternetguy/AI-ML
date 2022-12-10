import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

model.load_weights('model.h5')

# predict
predictions = model.predict(test_images[:5])

# print predictions
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# validate
print(test_labels[:5]) # [7, 2, 1, 0, 4]
