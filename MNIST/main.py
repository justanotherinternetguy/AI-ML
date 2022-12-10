import numpy as np
import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# normalize
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# flatten
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# build model structure
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# compile model
# lr = learning rate
model.compile(
  optimizer=Adam(learning_rate=0.005),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# train model
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=32,
)

# evaluate (check accuracy)
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# save weights
model.save_weights('model.h5')

# model.load_weights('model.h5')

# predict on new images
predictions = model.predict(test_images[:5])

print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# validate preds
print(test_labels[:5]) # [7, 2, 1, 0, 4]
