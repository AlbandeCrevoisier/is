import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop


# Get dataset
X_pos, X_neg, y_pos, y_neg = [], [], [], []
for f in os.listdir('data/pictures/with_pool'):
    X_pos.append(plt.imread('data/pictures/with_pool/' + f))
    y_pos.append(1)
X_pos, y_pos = np.array(X_pos), np.array(y_pos).reshape(-1, 1)
for f in os.listdir('data/pictures/without_pool'):
    X_neg.append(plt.imread('data/pictures/without_pool/' + f))
    y_neg.append(0)
X_neg, y_neg = np.array(X_neg), np.array(y_neg).reshape(-1, 1)

# Undersample to correct class imbalance: 2.87% of images contain a pool.
i = np.random.permutation(len(X_neg))
X_neg = X_neg[i[:len(X_pos)]]
y_neg = y_neg[i[:len(y_pos)]]

X = np.concatenate((X_pos, X_neg)) / 255
y = np.concatenate((y_pos, y_neg))

# Shuffle data to mix pos and neg
i = np.random.permutation(X.shape[0])
i_split = int(np.floor(0.8 * X.shape[0]))
X_train, X_test = X[i[:i_split]], X[i[i_split:]]
y_train, y_test = y[i[:i_split]], y[i[i_split:]]

train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.batch(10).repeat()
test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test = test.batch(10).repeat()

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                 input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])
model.fit(train, epochs=5, steps_per_epoch=10,
          validation_data=test, validation_steps=3)

