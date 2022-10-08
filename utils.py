from cv2 import COLOR_RGB2GRAY
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io as sio
import h5py
import imgvision as iv

'''
    一些工具
'''

def conv(filters, size, stride, activation, apply_instnorm=True):
    '''
        Conv2D 加 instance_normalization
    '''
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, stride, padding='same', use_bias=True))
    if apply_instnorm:
        result.add(tfa.layers.InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def conv_transp(filters, size, stride, activation, apply_instnorm=True):
    '''
        Conv2DTranspose 加 instance_normalization
    '''
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, stride, padding='same', use_bias=True))
    if apply_instnorm:
        result.add(tfa.layers.InstanceNormalization())
    if not activation == None: # 
        result.add(activation())
    return result

def CTIS_norm_crop2same(origin_img):
    '''
        将(512,512)的CTIS成像切割成(170,170,9)并返回
        PS: 这是用于测试将CTIS图像均分为9份的方法
    '''
    r,c = origin_img.shape
    sz = r//3
    img_0th = origin_img[sz:sz+sz, sz:sz+sz]
    img_1th_1 = origin_img[:sz,:sz]
    img_1th_2 = origin_img[:sz, sz:sz+sz]
    img_1th_3 = origin_img[:sz, sz+sz:sz+sz+sz]
    img_1th_4 = origin_img[sz:sz+sz, :sz]
    img_1th_5 = origin_img[sz:sz+sz, sz+sz:sz+sz+sz]
    img_1th_6 = origin_img[sz+sz:sz+sz+sz, :sz]
    img_1th_7 = origin_img[sz+sz:sz+sz+sz, sz:sz+sz]
    img_1th_8 = origin_img[sz+sz:sz+sz+sz, sz+sz:sz+sz+sz]

    img = np.concatenate(
        [img_0th[..., np.newaxis], \
        img_1th_1[...,np.newaxis], \
        img_1th_2[...,np.newaxis], \
        img_1th_3[...,np.newaxis], \
        img_1th_4[...,np.newaxis], \
        img_1th_5[...,np.newaxis], \
        img_1th_6[...,np.newaxis], \
        img_1th_7[...,np.newaxis], \
        img_1th_8[...,np.newaxis]], \
            axis=2)

    return img

def CTIS_norm_crop(origin_img, lefttop_0th=(224,224), width_0th=64, delta_0=64, delta_rec=159, delta_diag=128):
    '''
        (512, 512) => [(64,64), (64,159,4), (128,128,4)]
        根据CTIS simulator模拟生成的CTIS图像, 切割成GAN网络输入的三个部分
        注: 如果用于DOE真实情况模拟, 则需要重构代码
    '''
    
    # 0 order
    img_0th_lt_x = lefttop_0th[0]
    img_0th_lt_y = lefttop_0th[1]
    img_0th_rb_x = img_0th_lt_x + width_0th
    img_0th_rb_y = img_0th_lt_y + width_0th
    img_0th = origin_img[img_0th_lt_x:img_0th_rb_x, img_0th_lt_y:img_0th_rb_y]

    # 1 order rectangle
    # top
    img_1th_lt_1_x = img_0th_lt_x - delta_0 - delta_rec
    img_1th_lt_1_y = img_0th_lt_y
    img_1th_rb_1_x = img_1th_lt_1_x + delta_rec
    img_1th_rb_1_y = img_1th_lt_1_y + width_0th
    img_1th_1 = origin_img[img_1th_lt_1_x:img_1th_rb_1_x, img_1th_lt_1_y:img_1th_rb_1_y]
    # left
    img_1th_lt_2_x = img_0th_lt_x
    img_1th_lt_2_y = img_0th_lt_y - delta_0 - delta_rec
    img_1th_rb_2_x = img_1th_lt_2_x + width_0th
    img_1th_rb_2_y = img_1th_lt_2_y + delta_rec
    img_1th_2 = origin_img[img_1th_lt_2_x:img_1th_rb_2_x, img_1th_lt_2_y:img_1th_rb_2_y]
    # right
    img_1th_lt_3_x = img_0th_lt_x
    img_1th_lt_3_y = img_0th_lt_y + delta_0 + width_0th
    img_1th_rb_3_x = img_1th_lt_3_x + width_0th
    img_1th_rb_3_y = img_1th_lt_3_y + delta_rec
    img_1th_3 = origin_img[img_1th_lt_3_x:img_1th_rb_3_x, img_1th_lt_3_y:img_1th_rb_3_y]
    # bottom
    img_1th_lt_4_x = img_0th_lt_x + delta_0 + width_0th
    img_1th_lt_4_y = img_0th_lt_y
    img_1th_rb_4_x = img_1th_lt_4_x + delta_rec
    img_1th_rb_4_y = img_1th_lt_4_y + width_0th
    img_1th_4 = origin_img[img_1th_lt_4_x:img_1th_rb_4_x, img_1th_lt_4_y:img_1th_rb_4_y]
    # rotate & concatenate
    img_1th_1 = np.rot90(img_1th_1, 3)
    img_1th_2 = np.rot90(img_1th_2, 2)
    img_1th_4 = np.rot90(img_1th_4, 1)
    img_1th_rect = np.concatenate([img_1th_1[...,np.newaxis], img_1th_2[...,np.newaxis], \
        img_1th_3[...,np.newaxis], img_1th_4[...,np.newaxis]], axis=2)

    # 1 order diagonal
    # left top
    img_1th_lt_1_x = img_0th_lt_x - delta_0 - delta_diag
    img_1th_lt_1_y = img_0th_lt_y - delta_0 - delta_diag
    img_1th_rb_1_x = img_1th_lt_1_x + delta_diag
    img_1th_rb_1_y = img_1th_lt_1_y + delta_diag
    img_1th_1 = origin_img[img_1th_lt_1_x:img_1th_rb_1_x, img_1th_lt_1_y:img_1th_rb_1_y]
    # right top
    img_1th_lt_2_x = img_0th_lt_x - delta_0 - delta_diag
    img_1th_lt_2_y = img_0th_lt_y + delta_0 + width_0th
    img_1th_rb_2_x = img_1th_lt_2_x + delta_diag
    img_1th_rb_2_y = img_1th_lt_2_y + delta_diag
    img_1th_2 = origin_img[img_1th_lt_2_x:img_1th_rb_2_x, img_1th_lt_2_y:img_1th_rb_2_y]
    # left bottom
    img_1th_lt_3_x = img_0th_lt_x + width_0th + delta_0
    img_1th_lt_3_y = img_0th_lt_y - delta_0 - delta_diag
    img_1th_rb_3_x = img_1th_lt_3_x + delta_diag
    img_1th_rb_3_y = img_1th_lt_3_y + delta_diag
    img_1th_3 = origin_img[img_1th_lt_3_x:img_1th_rb_3_x, img_1th_lt_3_y:img_1th_rb_3_y]
    # right bottom
    img_1th_lt_4_x = img_0th_lt_x + width_0th + delta_0
    img_1th_lt_4_y = img_0th_lt_y + width_0th + delta_0
    img_1th_rb_4_x = img_1th_lt_4_x + delta_diag
    img_1th_rb_4_y = img_1th_lt_4_y + delta_diag
    img_1th_4 = origin_img[img_1th_lt_4_x:img_1th_rb_4_x, img_1th_lt_4_y:img_1th_rb_4_y]
    # rotate & concatenate
    img_1th_1 = np.rot90(img_1th_1, 3)
    img_1th_3 = np.rot90(img_1th_3, 2)
    img_1th_4 = np.rot90(img_1th_4, 1)
    img_1th_diag = np.concatenate([img_1th_1[...,np.newaxis], img_1th_2[...,np.newaxis], \
        img_1th_3[...,np.newaxis], img_1th_4[...,np.newaxis]], axis=2)

    return [img_0th, img_1th_rect, img_1th_diag]


def get_all_inputs(dir='./MIXED_CTIS_norm'):
    '''
        得到所有的输入 (size, 512, 512)

        dir: 输入文件的目录
    '''
    data_input_dir = dir 
    data_input = []

    for i,e in enumerate(sorted(os.listdir(data_input_dir))):
        tp = cv2.imread(data_input_dir + '/' + e)
        tp = rgb2gray_01(tp)
        data_input.append(tp)

    data_input = np.array(data_input)
    
    return data_input


def get_all_outputs(dir='./MIXED_dataset_64_64_31'):
    '''
        得到所有的输出 (size, 64, 64, 31)

        dir: 输出文件的目录
    '''
    data_output_dir = dir
    data_output = []

    for i,e in enumerate(sorted(os.listdir(data_output_dir))):
        tp = np.load(data_output_dir + '/' + e)
        data_output.append(tp)
        
    data_output = np.array(data_output)
    
    return data_output


def get_all_single_inputs():
    '''
        针对 single 的测试
        得到所有的输入 (size, 512, 512)
    '''
    data_input_dir = './single_CTIS_norm' 
    data_input = []

    for i,e in enumerate(sorted(os.listdir(data_input_dir))):
        tp = cv2.imread(data_input_dir + '/' + e)
        tp = rgb2gray_01(tp)
        data_input.append(tp)

    data_input = np.array(data_input)
    
    return data_input


def get_all_single_outputs():
    '''
        针对 single 的测试
        得到所有的输出 (size, 64, 64, 31)
    '''
    data_output_dir = './single_dataset_64_64_31'
    data_output = []

    for i,e in enumerate(sorted(os.listdir(data_output_dir))):
        tp = np.load(data_output_dir + '/' + e)
        data_output.append(tp)
        
    data_output = np.array(data_output)
    
    return data_output


def input2ginput(batch_input):
    '''
        将 (bs, 512, 512) 的输入 转化为 [(bs, 64, 64, 1), (bs, 64, 159, 4), (bs, 128, 128, 4)] 的输入
    '''
    # bs,r,c = batch_input.shape
    img_0th, img_1th_rect, img_1th_diag = [], [], []
    for i in range(batch_input.shape[0]):
        t1, t2, t3 = CTIS_norm_crop(batch_input[i,...])
        img_0th.append(t1)
        img_1th_rect.append(t2)
        img_1th_diag.append(t3)
    img_0th = np.array(img_0th)
    img_0th = tf.cast(img_0th[..., np.newaxis], dtype=tf.float32)
    img_1th_rect = tf.cast(img_1th_rect, dtype=tf.float32)
    img_1th_diag = tf.cast(img_1th_diag, dtype=tf.float32)
    
    return [img_0th, img_1th_rect, img_1th_diag]


def input2ginput_same(batch_input):
    '''
        将 (bs, 512, 512) 的输入 转化为 [(bs, 170, 170, 9)] 的输入
        PS: 这是用于测试将CTIS图像均分为9份的方法
    '''
    # bs,r,c = batch_input.shape
    img = []
    for i in range(batch_input.shape[0]):
        t = CTIS_norm_crop2same(batch_input[i,...])
        img.append(t)
    img = tf.cast(img, dtype=tf.float32)

    return img
    

def l2_loss(y, recon):
    '''
        L2 loss
    '''
    return tf.reduce_mean(tf.losses.MSE(y, recon))

def l1_loss(y, recon):
    '''
        L1 loss
    '''
    return tf.reduce_mean(tf.losses.MAE(y, recon))


def ssim_loss(y, recon):
    '''
        SSIM loss
    '''
    y = tf.cast(y, dtype=tf.double)
    recon = tf.cast(recon, dtype=tf.double)
    return 1 - tf.reduce_mean(tf.image.ssim(y, recon, max_val=1.0))


def content_loss(y, recon, fe_model):
    '''
        content loss
        fe_model: feature extraction model (e.g. VGG19)
        PS: 没有写完，实际没有用到
    '''
    y = fe_model.predict(y)
    recon = fe_model.predict(recon)
    return l2_loss(y, recon)


# .npy 转为 .mat 并保存
def npy2mat(npyfile_name, matfile_name='./res.mat', dicname='img', save=True):
    try:
        imgmat = np.load(npyfile_name)
        if save:
            sio.savemat(matfile_name,{dicname: imgmat})
        return imgmat
    except:
        raise(ValueError("file not correct"))


# 图片 normalize
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


# RGB图转灰度图 numpy 格式
def rgb2gray_01(img):
    return cv2.cvtColor(img, COLOR_RGB2GRAY).astype(np.float32) / 255.0


# 图片缩放
def resize_img(img, size, method='bilinear'):
    return tf.image.resize(img[:,:,np.newaxis], size=size, method=method)[:,:,0]


# .mat 转为 .npy 并保存
def mat2npy(matfile_name, npyfile_name='./res.npy', dicname='rad', save=True):
    try:
        img = h5py.File(matfile_name)[dicname]
        if save:
            np.save(npyfile_name, img)
        return img
    except:
        raise(ValueError("file not correct"))


# (X, Y, 31) => color image
def HSI2color(img):
    convertor = iv.spectra(illuminant='D65',band = np.arange(400,710,10))
    return convertor.space(img, 'srgb')


# 01 image => uint8
def to_uint8(img):
    return np.array(img * 255).astype(np.uint8)


# 将一个npy文件保存为(npy, png, mat)格式
def savenpy_as_N_P_M(npyimg, savedir: str, savename: str, dicname='img'):
    npyimg = np.maximum(npyimg, 0)
    npyimg = np.minimum(npyimg, 1)
    if len(npyimg.shape) == 3: npyimg = npyimg[...,::-1]
    np.save(f'{savedir}/{savename}.npy', npyimg)
    cv2.imwrite(f'{savedir}/{savename}.png', to_uint8(npyimg))
    sio.savemat(f'{savedir}/{savename}.mat', {dicname: npyimg})
    print(f'save [{savename}] to [{savedir}] done')



def get_test_io(dir='./paper_content/images'):
    '''
        获取test的数据
        return (data_input, data_output)
    '''
    data_dir = dir 
    data_input = []
    data_output = []

    for i,e in enumerate(sorted(os.listdir(data_dir))):
        suf = e[:-4].split('_')[-1]
        if suf == 'input':
            tp = cv2.imread(data_dir + '/' + e)
            tp = rgb2gray_01(tp)
            data_input.append(tp)
        elif suf == 'hsi':
            tp = np.load(data_dir + '/' + e)
            data_output.append(tp)

    data_input = np.array(data_input)
    data_output = np.array(data_output)
    
    return data_input, data_output


def psnr(rec_img, real_img):
    '''
        calculate PSNR of a multi channel image
    '''
    real_img = tf.cast(real_img, dtype=tf.double)
    rec_img = tf.cast(rec_img, dtype=tf.double)
    return tf.image.psnr(real_img, rec_img, max_val=1.0)


def ssim(rec_img, real_img):
    '''
        calculate SSIM of a multi channel image
    '''
    real_img = tf.cast(real_img, dtype=tf.double)
    rec_img = tf.cast(rec_img, dtype=tf.double)
    return tf.image.ssim(real_img, rec_img, max_val=1.0)


def mean_filter(img, ksize):
    '''
        均值滤波
    '''
    return cv2.blur(img, (ksize,ksize))


def GIF_filter(img, guided_img, ksize):
    '''
        引导图像滤波 guided image filtering
    '''
    mean_I = mean_filter(guided_img, ksize)
    mean_p = mean_filter(img, ksize)
    corr_I = mean_filter(guided_img * guided_img, ksize)
    corr_Ip = mean_filter(guided_img * img, ksize)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + 0.001)
    b = mean_p - a * mean_I
    mean_a = mean_filter(a, ksize)
    mean_b = mean_filter(b, ksize)
    q = mean_a * guided_img + mean_b
    q[q<0] = 0
    return q

if __name__ == '__main__':

    # img = cv2.imread("./CTIS_norm/001.png")
    # img = rgb2gray_01(img)
    # ans = CTIS_norm_crop2same(img)
    # print(ans.shape)


    # for i in range(9):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(ans[...,i], cmap='gray')
    # plt.show()
    
    a = tf.random.normal([2, 512, 512])
    a = input2ginput_same(a)
    

    pass

