#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# 1. 自定义网络层
# 定义网络层就是：设置网络权重和输出到输入的计算过程

class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()
        
        # w_init = tf.random_normal_initializer()
        # self.weight = tf.Variable(initial_value=w_init(
        #     shape=(input_dim, unit), dtype=tf.float32), trainable=True)
        
        # b_init = tf.zeros_initializer()
        # self.bias = tf.Variable(initial_value=b_init(
        #     shape=(unit,), dtype=tf.float32), trainable=True)

        # 当然我们也可以直接用add_weight的方法构建权重
        self.weight = self.add_weight(shape=(input_dim, unit),
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True) # trainable=False就为不可训练权重
        self.bias = self.add_weight(shape=(unit,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=True) # trainable=False就为不可训练权重
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

x = tf.ones((3,5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)



# 2. 使用子层递归构建网络层
class MyBlock(layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)
    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)
    
my_block = MyBlock()
print('trainable weights:', len(my_block.trainable_weights))
y = my_block(tf.ones(shape=(3, 64)))
# 构建网络在build()里面，所以执行了才有网络
print('trainable weights:', len(my_block.trainable_weights)) 




# 3. 其他网络层配置
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
​
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self): # 使自己的网络层可以序列化，继承Linear
        config = super(Linear, self).get_config()
        config.update({'units':self.units})
        return config
    
layer = Linear(125)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)



# 4. 构建自己的网络
# 采样网络
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
# 编码器
class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
        
# 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim, 
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
    
# 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
    
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        kl_loss = -0.5*tf.reduce_sum(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed
​
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
vae = VAE(784,32,64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

# 编写训练方法
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
​
original_dim = 784
vae = VAE(original_dim, 64, 32)
​
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
​
loss_metric = tf.keras.metrics.Mean()
​
# 每个epoch迭代.
for epoch in range(3):
  print('Start of epoch %d' % (epoch,))
​
  # 取出每个batch的数据并训练.
  for step, x_batch_train in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      reconstructed = vae(x_batch_train)
      # 计算 reconstruction loss
      loss = mse_loss_fn(x_batch_train, reconstructed)
      loss += sum(vae.losses)  # 添加 KLD regularization loss
      
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    
    loss_metric(loss)
    
    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, loss_metric.result()))