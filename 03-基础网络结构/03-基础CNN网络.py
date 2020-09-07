#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
plt.imshow(x_train[0])
plt.show()

# 2. 构造网络
x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))
model = keras.Sequential()
# 添加卷积层
model.add(
  layers.Conv2D(
    input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
    filters=32, 
    kernel_size=(3,3), 
    strides=(1,1), 
    padding='valid',
    activation='relu'
  )
)
# 添加池化层
model.add(layers.MaxPool2D(pool_size=(2,2)))
# 添加全连接层
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 3. 模型配置与训练
model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
res = model.evaluate(x_test, y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()