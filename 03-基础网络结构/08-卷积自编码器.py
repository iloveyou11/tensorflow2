#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train.astype('float32'), -1) / 255.0
x_test = tf.expand_dims(x_test.astype('float32'),-1) / 255.0


# 2. 构造网络
inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='inputs')
code = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
code = layers.MaxPool2D((2,2), padding='same')(code)
decoded = layers.Conv2D(16, (3,3), activation='relu', padding='same')(code)
decoded = layers.UpSampling2D((2,2))(decoded)
outputs = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(decoded)
auto_encoder = keras.Model(inputs, outputs)
auto_encoder.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.BinaryCrossentropy())
keras.utils.plot_model(auto_encoder, show_shapes=True)


# 3. 训练与测试
early_stop = keras.callbacks.EarlyStopping(patience=2, monitor='loss')
auto_encoder.fit(x_train,x_train, batch_size=64, epochs=1, validation_split=0.1,validation_freq=10,
                callbacks=[early_stop])
decoded = auto_encoder.predict(x_test)

n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(tf.reshape(x_test[i+1],(28, 28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(tf.reshape(decoded[i+1],(28, 28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()