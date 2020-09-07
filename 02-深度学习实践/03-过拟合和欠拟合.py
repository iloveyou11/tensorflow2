#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 防止神经网络中过度拟合的最常用方法：
# 获取更多训练数据。
# 减少网络容量。
# 添加权重正规化。
# 添加dropout。



# 1. 加载数据
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# 热编码
def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
# plt.plot(train_data[0]) # 打印第一个文本的独热编码


# 2. 创建baseline模型
# baseline_model = keras.Sequential([
#     layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# baseline_model.compile(optimizer='adam',
#                       loss='binary_crossentropy',
#                       metrics=['accuracy', 'binary_crossentropy'])
# baseline_model.summary()
# baseline_history = baseline_model.fit(train_data, train_labels,
#                                      epochs=20, batch_size=512,
#                                      validation_data=(test_data, test_labels),
#                                      verbose=2)



# 3. 创建一个小模型
# 注意这里网络层神经元的个数 4 4 1
small_model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
small_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
small_model.summary()
small_history = small_model.fit(train_data, train_labels,
                                     epochs=20, batch_size=512,
                                     validation_data=(test_data, test_labels),
                                     verbose=2)



# 4. 创建一个大模型 512 512 1
# 注意这里网络层神经元的个数
big_model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
big_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
big_model.summary()
big_history = big_model.fit(train_data, train_labels,
                                     epochs=20, batch_size=512,
                                     validation_data=(test_data, test_labels),
                                     verbose=2)



# 5. 绘制图
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()
  plt.xlim([0,max(history.epoch)])
plot_history([('baseline', baseline_history),
              ('small', small_history),
              ('big', big_history)])


# 6. 增加L2正则
l2_model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), 
                 activation='relu', input_shape=(NUM_WORDS,)),
    layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), 
                 activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
l2_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
l2_model.summary()
l2_history = l2_model.fit(train_data, train_labels,
                                     epochs=20, batch_size=512,
                                     validation_data=(test_data, test_labels),
                                     verbose=2)
plot_history([('baseline', baseline_history),
              ('l2', l2_history)])


# 7. 添加dropout
dpt_model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
dpt_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
dpt_model.summary()
dpt_history = dpt_model.fit(train_data, train_labels,
                                     epochs=20, batch_size=512,
                                     validation_data=(test_data, test_labels),
                                     verbose=2)
plot_history([('baseline', baseline_history),
              ('dropout', dpt_history)])