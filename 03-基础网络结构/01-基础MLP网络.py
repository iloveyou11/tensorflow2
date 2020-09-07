#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


# 1. 回归任务
# 导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
model = keras.Sequential([
    layers.Dense(32, activation='sigmoid', input_shape=(13,)),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(1)
])
model.summary()
# 配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),
             loss='mean_squared_error', 
             metrics=['mse'])
# 训练
model.fit(x_train, y_train, batch_size=50, epochs=50, validation_split=0.1, verbose=1)
result = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(result)


# 2. 分类任务
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=7)

# 构建模型
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(30,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.binary_crossentropy,
             metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test)
print(model.metrics_names)