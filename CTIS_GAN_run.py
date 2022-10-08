# %%
import os
import glob
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, optimizers
from utils import CTIS_norm_crop, rgb2gray_01, get_all_inputs, get_all_outputs, input2ginput
from utils import l1_loss, ssim_loss, l2_loss
from CTIS_GAN_GD import Generator, Discriminator
tf.random.set_seed(666666)

# %% [markdown]
# # CTIS-GAN

# %%
def slice_train_test(batch_size=1, shuffle_size=10, train_ratio=0.9):
    '''
        切分图片为训练集和测试集
    '''
    
    data_input = get_all_inputs()
    data_output = get_all_outputs()

    # print(data_input.shape, data_output.shape)

    train_len = int(train_ratio * data_input.shape[0])
    test_len = data_input.shape[0] - train_len

    x_train = data_input[:train_len,...]
    y_train = data_output[:train_len,...]
    x_test = data_input[-test_len:,...]
    y_test = data_output[-test_len:,...]
    db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    db_train = db_train.shuffle(shuffle_size).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    db_test = db_test.shuffle(shuffle_size).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return db_train, db_test
    # return db_train

# %%
def gradient_penalty(discriminator, batch_real, batch_fake, condition):
    '''
        gradient penalty
    '''
    
    batch_real = tf.cast(batch_real, dtype=tf.float32)
    batch_fake = tf.cast(batch_fake, dtype=tf.float32)
    batchsz = batch_real.shape[0]

    t = tf.random.uniform([batchsz, 1, 1, 1])
    t = tf.broadcast_to(t, batch_real.shape) # 将随机权重broadcast成输入的形状

    interpolate = t * batch_real + (1 - t) * batch_fake # 直接线性插值

    with tf.GradientTape() as tape:
        tape.watch([interpolate])
        d_interpolate_logits = discriminator([condition, interpolate])
    grads = tape.gradient(d_interpolate_logits, interpolate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp-1)**2)

    return gp


# %%
def celoss(logits, flag):
    '''
        cross entropy loss
        flag: 'ones' or 'zeros'
    '''
    if flag == 'ones':
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    elif flag == 'zeros':
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_input, batch_output, is_training: bool):
    '''
        计算 discriminator 的 loss function
    '''
    img_0th, img_1th_rect, img_1th_diag = input2ginput(batch_input=batch_input)
    fake_image = generator([img_0th, img_1th_rect, img_1th_diag], is_training)

    d_fake_logits = discriminator([img_0th, fake_image], is_training)
    d_real_logits = discriminator([img_0th, batch_output], is_training)

    d_loss_fake = celoss(d_fake_logits, 'zeros') # discriminator loss with fake
    d_loss_real = celoss(d_real_logits, 'ones') # discriminator loss with real

    # calculate gradient penalty
    gp = gradient_penalty(discriminator, batch_output, fake_image, img_0th)
    alpha = 10.0 # gradiant penalty weight

    # discriminator loss
    loss = d_loss_fake + d_loss_real + alpha * gp

    return loss, gp

def g_loss_fn(generator, discriminator, batch_input, batch_output, is_training: bool):
    '''
        计算 generator 的 loss function
    '''

    img_0th, img_1th_rect, img_1th_diag = input2ginput(batch_input=batch_input)
    fake_image = generator([img_0th, img_1th_rect, img_1th_diag], is_training)

    d_fake_logits = discriminator([img_0th, fake_image], is_training)
    d_loss_fake = celoss(d_fake_logits, 'ones')

    g_ssim_loss = ssim_loss(batch_output, fake_image)
    g_l1_loss = l1_loss(batch_output, fake_image)
    g_l2_loss = l2_loss(batch_output, fake_image)
    ssim_w = 1 # ssim loss weight
    l1_w = 1 # l1 loss weight
    l2_w = 1 # l2 loss weight
    # spatial loss = sum(weight_i * loss_i)
    spatial_loss = (ssim_w * g_ssim_loss) + (l1_w * g_l1_loss) + (l2_w * g_l2_loss)
    
    
    d_w = 1 # discriminator loss weight
    spatial_w = 100 # spatial loss weight
    # generator loss = sum(weight_i * loss_i)
    loss = (d_w * d_loss_fake) + (spatial_w * spatial_loss)

    return loss


generator = Generator() # 生成generater实例对象
discriminator = Discriminator() # 生成discriminator实例对象
lr_g = 0.0001 # 设置generator的学习率
lr_d = 0.0001 # 设置discriminator的学习率
g_optimizer = tf.optimizers.Adam(learning_rate=lr_g, beta_1=0.5) # 设置generator优化函数
d_optimizer = tf.optimizers.Adam(learning_rate=lr_d, beta_1=0.5) # 设置discriminator优化函数
epoches = 30000 # 迭代epoch设定
batch_size = 32 # 设置batch size
shuffle_size = 6000 # 设置shuffle size
train_ratio = 0.9 # 设置训练和测试数据的比例
db_train, db_test = slice_train_test(batch_size=batch_size, shuffle_size=shuffle_size, train_ratio=train_ratio)

# set checkpoint 设定模型保存
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer, generator=generator, \
    d_optimizer=d_optimizer, discriminator=discriminator)
save_dir = './ckpt_CTIS_GAN/'
manager = tf.train.CheckpointManager(checkpoint, directory=save_dir,  max_to_keep=2)

d_loss_ls = [] # 用于存discriminator的loss
g_loss_ls = [] # 用于村generator的loss
gp_ls = [] # 用于村gradient penalty
for epoch in range(epoches):
    batch_input, batch_output = next(iter(db_train))
    # print(batch_input.shape,batch_output.shape)
    # break

    ### train discriminator ###
    with tf.GradientTape() as tape:
        d_loss, gp = d_loss_fn(generator, discriminator, batch_input, batch_output, is_training=True)
    grads = tape.gradient(d_loss, discriminator.trainable_variables) # get gradients
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables)) # apply gradients through optimizer
    # print(len(discriminator.trainable_variables))

    ### train generator ###
    with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_input, batch_output, is_training=True)
    grads = tape.gradient(g_loss, generator.trainable_variables) # same as above
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables)) # same as above
    # print(len(generator.trainable_variables))

    print(f"epoch:{epoch}  d_loss:{d_loss}  g_loss:{g_loss}  gp:{gp}") # log
    d_loss_ls.append(d_loss)
    g_loss_ls.append(g_loss)
    gp_ls.append(gp)
    if (epoch+1)%100 == 0: # checkpoint save
        path = manager.save()
        print(f"model saved to {path}")
        with open('./d_loss.txt','a') as f:
            f.writelines(f'{" ".join(list(map(str, d_loss_ls)))} ')
            d_loss_ls = []
        with open('./gp.txt','a') as f:
            f.writelines(f'{" ".join(list(map(str, gp_ls)))} ')
            gp_ls = []
        with open('./g_loss.txt','a') as f:
            f.writelines(f'{" ".join(list(map(str, g_loss_ls)))} ')
            g_loss_ls = []