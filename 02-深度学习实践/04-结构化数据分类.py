#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 如何对结构化数据进行分类（例如CSV中的表格数据）?
# 包含了以下几个步骤：
# 1. 使用Pandas加载CSV文件。
# 2. 构建一个输入的pipeline，使用tf.data批处理和打乱数据。
# 3. 从CSV中的列映射到用于训练模型的输入要素。
# 4. 使用Keras构建，训练和评估模型。


# 1. 数据加载
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# 使用tf.data构造输入pipeline
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )


# tensorflow的feature column
example_batch = next(iter(train_ds))[0]
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())
# 数字列
age = tf.feature_column.numeric_column("age")
demo(age)
[[61.]
 [51.]
 [57.]
 [51.]
 [44.]]
# Bucketized列（桶列）
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[
    18, 25, 30, 35, 40, 50
])
demo(age_buckets)
# 类别列
thal = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = tf.feature_column.indicator_column(thal)
demo(thal_one_hot)
# 嵌入列
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)
# 哈希特征列
thal_hashed = tf.feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(thal_hashed))
# 交叉功能列
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(crossed_feature))



# 2. 选择使用feature column
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = tf.feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# 构建特征层
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)



# 3. 构建模型并训练
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds,epochs=5)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)