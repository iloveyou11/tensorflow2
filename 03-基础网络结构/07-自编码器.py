#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import SVG


# 1. 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test = x_test.reshape((-1, 28*28)) / 255.0


# 2. 定义自编码器
code_dim = 32
inputs = layers.Input(shape=(x_train.shape[1],), name='inputs')
code = layers.Dense(code_dim, activation='relu', name='code')(inputs)
outputs = layers.Dense(x_train.shape[1], activation='softmax', name='outputs')(code)
auto_encoder = keras.Model(inputs, outputs)
encoder = keras.Model(inputs,code)
decoder_input = keras.Input((code_dim,))
decoder_output = auto_encoder.layers[-1](decoder_input)
decoder = keras.Model(decoder_input, decoder_output)

auto_encoder.summary()

keras.utils.plot_model(auto_encoder, show_shapes=True)
keras.utils.plot_model(encoder, show_shapes=True)
keras.utils.plot_model(decoder, show_shapes=True)

auto_encoder.compile(optimizer='adam',loss='binary_crossentropy')


# 3. 训练模型
history = auto_encoder.fit(x_train, x_train, batch_size=64, epochs=100, validation_split=0.1)
encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)

plt.figure(figsize=(10,4))
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(decoded[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()