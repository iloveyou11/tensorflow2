#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from tensorflow import keras
from tensorflow.keras import layers



# 使用决策树和tf.estimator API训练Gradient Boosting模型的端到端演练



# 1. 加载数据

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
import tensorflow as tf
tf.random.set_seed(123)
# print(dftrain.head())
# print(dftrain.describe())



# 2. 查看数据
dftrain.shape[0], dfeval.shape[0] # 训练集，验证集数量
(627, 264)
# 年龄分布
dftrain.age.hist(bins=20)
# 男女比例
dftrain.sex.value_counts().plot(kind='barh')
# 大部分为三等顾客
dftrain['class'].value_counts().plot(kind='barh')
# 大多数乘客从南安普敦出发。
dftrain['embark_town'].value_counts().plot(kind='barh')
# 与男性相比，女性存活的机率要高得多。 这显然是该模型的预测特征
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
Text(0.5, 0, '% survive')



# 3. 构造输入特征
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,type=tf.float32))

example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()


# 构造输入数据.
NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)



# 4. 训练和验证模型
linear_est = tf.estimator.LinearClassifier(feature_columns)

# 训练
linear_est.train(train_input_fn, max_steps=100)

# 验证
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))


n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)
# 训练
est.train(train_input_fn, max_steps=100)
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))


# 用训练好的模型进行预测
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')



# 5. 观察roc得分
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)