import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop


X, y = [], []
for f in os.listdir('data/pictures/with_pool'):
    X.append(plt.imread('data/pictures/with_pool/' + f))
    y.append(1)
for f in os.listdir('data/pictures/without_pool'):
    X.append(plt.imread('data/pictures/without_pool/' + f))
    y.append(0)
X = np.array(X)
y = np.array(y).reshape(len(y), 1)

i = np.random.permutation(X.shape[0])
i_split = int(np.floor(0.8 * X.shape[0]))
X_train, X_test = X[i[:i_split]], X[i[i_split:]]
y_train, y_test = y[i[:i_split]], y[i[i_split:]]

train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.batch(32).repeat()
test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test = test.batch(32).repeat()

model = Sequential()
model.add(Conv2D(8, (3, 3), padding='same', activation='relu',
                 input_shape=(256, 256, 3)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001, decay=1e-6),
              metrics=['binary_accuracy'])
model.fit(train, epochs=5, steps_per_epoch=10,
          validation_data=test, validation_steps=2)

