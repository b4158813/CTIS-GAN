from re import S
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class Generator(keras.Model):
    '''
        GAN_1
    '''
    def __init__(self):
        super(Generator, self).__init__()

        # generator parameter

        ################ encoder ###################
        # deal with the 0th order #
        self.conv_c_1_1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_1_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_c_1_2 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_1_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_c_1_3 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_1_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_c_1_d = layers.Conv2D(64, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_c_1_d = layers.LeakyReLU(alpha=0.2)
        self.conv_c_2_1 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_2_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_c_2_2 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_2_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_c_2_3 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_c_2_3 = layers.LeakyReLU(alpha=0.2)
        # deal with the rectangle 1st order #
        self.conv_r_1_1 = layers.Conv2D(32, kernel_size=1, strides=(1,1), padding='same')
        self.LReLU_r_1_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_r_2_1 = layers.Conv2D(32, kernel_size=3, strides=(2,5), padding='same')
        self.LReLU_r_2_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_r_2_2 = layers.Conv2D(32, kernel_size=5, strides=(2,5), padding='same')
        self.LReLU_r_2_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_r_2_3 = layers.Conv2D(32, kernel_size=7, strides=(2,5), padding='same')
        self.LReLU_r_2_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_r_3_1 = layers.Conv2D(128, kernel_size=1, strides=(1,1), padding='same')
        self.LReLU_r_3_1 = layers.LeakyReLU(alpha=0.2)
        # deal with the diagonal 1st order #
        self.conv_d_1_1 = layers.Conv2D(32, kernel_size=1, strides=(1,1), padding='same')
        self.LReLU_d_1_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_2_1 = layers.Conv2D(32, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_2_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_2_2 = layers.Conv2D(32, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_2_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_2_3 = layers.Conv2D(32, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_2_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_2_d = layers.Conv2D(32, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_d_2_d = layers.LeakyReLU(alpha=0.2)
        self.conv_d_3_1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_3_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_3_2 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_3_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_3_3 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_3_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_3_d = layers.Conv2D(64, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_d_3_d = layers.LeakyReLU(alpha=0.2)
        self.conv_d_4_1 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_4_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_4_2 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_4_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_d_4_3 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_d_4_3 = layers.LeakyReLU(alpha=0.2)
        # deal with the concat above #
        self.conv_crd_1_1 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_1_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_1_2 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_1_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_1_3 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_1_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_1_d = layers.Conv2D(128, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_crd_1_d = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_2_1 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_2_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_2_2 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_2_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_2_3 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_2_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_2_d = layers.Conv2D(256, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_crd_2_d = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_3_1 = layers.Conv2D(512, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_3_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_3_2 = layers.Conv2D(512, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_3_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_crd_3_3 = layers.Conv2D(512, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_crd_3_3 = layers.LeakyReLU(alpha=0.2)

        ################ decoder ###################
        self.convtransp_drc_1_1 = layers.Conv2DTranspose(256, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_drc_1_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_2_1 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_2_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_2_2 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_2_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_2_3 = layers.Conv2D(256, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_2_3 = layers.LeakyReLU(alpha=0.2)
        self.convtransp_drc_3_1 = layers.Conv2DTranspose(128, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_drc_3_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_4_1 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_4_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_4_2 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_4_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_4_3 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_4_3 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_5_1 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_5_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_5_2 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_5_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_5_3 = layers.Conv2D(128, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_5_3 = layers.LeakyReLU(alpha=0.2)
        # attention parameter
        self.ch = 128 # 输入的通道数
        self.gamma = tf.Variable(initial_value=0.0, trainable=True)
        self.att_conv_theta = layers.Conv2D(self.ch//8, 1, (1,1), 'same')
        self.att_conv_phi = layers.Conv2D(self.ch//8, 1, (1,1), 'same')
        self.att_conv_g = layers.Conv2D(self.ch, 1, (1,1), 'same')
        ####
        self.convtransp_drc_6_1 = layers.Conv2DTranspose(64, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_drc_6_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_7_1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_7_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_7_2 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_7_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_drc_7_3 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_drc_7_3 = layers.LeakyReLU(alpha=0.2)

        self.conv_out_8 = layers.Conv2D(8, kernel_size=3, strides=(1,1), padding='same', activation=tf.nn.sigmoid)
        self.conv_add_1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_add_1 = layers.LeakyReLU(alpha=0.2)
        self.conv_out_16 = layers.Conv2D(16, kernel_size=3, strides=(1,1), padding='same', activation=tf.nn.sigmoid)
        self.conv_add_2 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='same')
        self.LReLU_add_2 = layers.LeakyReLU(alpha=0.2)
        self.conv_out_31 = layers.Conv2D(31,kernel_size=1, strides=(1,1), padding='same', activation=tf.nn.sigmoid)


    def attention(self, inputs):
        '''
            attention  layer
        '''
        bs,_,_,ch = inputs.shape # (bs, n, n, ch)
        theta = self.att_conv_theta(inputs)
        phi = self.att_conv_phi(inputs)
        g = self.att_conv_g(inputs)
        # theta * phi.T
        tp = tf.matmul(tf.reshape(theta, shape=[bs, -1, ch//8]), \
            tf.reshape(phi, shape=[bs, -1, ch//8]), transpose_b=True)
        # softmax
        att_map = tf.nn.softmax(tp, axis=-1)
        # attention_map * g
        out = tf.matmul(att_map, tf.reshape(g, shape=[bs, -1, ch]))
        out = tf.reshape(out, shape=inputs.shape)
        # γ * out + inputs
        outputs = self.gamma * out + inputs
        return outputs

    def call(self, inputs: list, training=None):
        c1, r1, d1 = inputs
        ########### encoder ############
        c2 = self.conv_c_1_1(c1)
        c2 = self.LReLU_c_1_1(c2)
        c2 = self.conv_c_1_2(c2)
        c2 = self.LReLU_c_1_2(c2)
        c2 = self.conv_c_1_3(c2)
        c2 = self.LReLU_c_1_3(c2)
        c3 = self.conv_c_1_d(c2) # down sample
        c3 = self.LReLU_c_1_d(c3)
        c4 = self.conv_c_2_1(c3)
        c4 = self.LReLU_c_2_1(c4)
        c4 = self.conv_c_2_2(c4)
        c4 = self.LReLU_c_2_2(c4)
        c4 = self.conv_c_2_3(c4)
        c4 = self.LReLU_c_2_3(c4)

        r2 = self.conv_r_1_1(r1)
        r2 = self.LReLU_r_1_1(r2)
        r31 = self.conv_r_2_1(r2)
        r31 = self.LReLU_r_2_1(r31)
        r32 = self.conv_r_2_2(r2)
        r32 = self.LReLU_r_2_2(r32)
        r33 = self.conv_r_2_3(r2)
        r33 = self.LReLU_r_2_3(r33)
        r4 = tf.concat([r31,r32,r33], axis=-1)
        r5 = self.conv_r_3_1(r4)
        r5 = self.LReLU_r_3_1(r5)
        
        d2 = self.conv_d_1_1(d1)
        d2 = self.LReLU_d_1_1(d2)
        d3 = self.conv_d_2_1(d2)
        d3 = self.LReLU_d_2_1(d3)
        d3 = self.conv_d_2_2(d3)
        d3 = self.LReLU_d_2_2(d3)
        d3 = self.conv_d_2_3(d3)
        d3 = self.LReLU_d_2_3(d3)
        d4 = self.conv_d_2_d(d3) # down sample
        d4 = self.LReLU_d_2_d(d4)
        d5 = self.conv_d_3_1(d4)
        d5 = self.LReLU_d_3_1(d5)
        d5 = self.conv_d_3_2(d5)
        d5 = self.LReLU_d_3_2(d5)
        d5 = self.conv_d_3_3(d5)
        d5 = self.LReLU_d_3_3(d5)
        d6 = self.conv_d_3_d(d5) # down sample
        d6 = self.LReLU_d_3_d(d6)
        d7 = self.conv_d_4_1(d6)
        d7 = self.LReLU_d_4_1(d7)
        d7 = self.conv_d_4_2(d7)
        d7 = self.LReLU_d_4_2(d7)
        d7 = self.conv_d_4_3(d7)
        d7 = self.LReLU_d_4_3(d7)

        crd1 = tf.concat([c4, r5, d7], axis=-1)
        crd2 = self.conv_crd_1_1(crd1)
        crd2 = self.LReLU_crd_1_1(crd2)
        crd2 = self.conv_crd_1_2(crd2)
        crd2 = self.LReLU_crd_1_2(crd2)
        crd2 = self.conv_crd_1_3(crd2)
        crd2 = self.LReLU_crd_1_3(crd2)
        crd3 = self.conv_crd_1_d(crd2) # down sample
        crd3 = self.LReLU_crd_1_d(crd3)
        crd4 = self.conv_crd_2_1(crd3)
        crd4 = self.LReLU_crd_2_1(crd4)
        crd4 = self.conv_crd_2_2(crd4)
        crd4 = self.LReLU_crd_2_2(crd4)
        crd4 = self.conv_crd_2_3(crd4)
        crd4 = self.LReLU_crd_2_3(crd4)
        crd5 = self.conv_crd_2_d(crd4) # down sample
        crd5 = self.LReLU_crd_2_d(crd5)
        crd6 = self.conv_crd_3_1(crd5)
        crd6 = self.LReLU_crd_3_1(crd6)
        crd6 = self.conv_crd_3_2(crd6)
        crd6 = self.LReLU_crd_3_2(crd6)
        crd6 = self.conv_crd_3_3(crd6)
        crd6 = self.LReLU_crd_3_3(crd6)


        ########### decoder ############
        drc1 = self.convtransp_drc_1_1(crd6)
        drc1 = self.LReLU_drc_1_1(drc1)
        drc1 = tf.concat([drc1, crd4], axis=-1)
        drc2 = self.conv_drc_2_1(drc1)
        drc2 = self.LReLU_drc_2_1(drc2)
        drc2 = self.conv_drc_2_2(drc2)
        drc2 = self.LReLU_drc_2_2(drc2)
        drc2 = self.conv_drc_2_3(drc2)
        drc2 = self.LReLU_drc_2_3(drc2)
        drc3 = self.convtransp_drc_3_1(drc2)
        drc3 = self.LReLU_drc_3_1(drc3)
        drc3 = tf.concat([drc3, crd2], axis=-1)
        drc4 = self.conv_drc_4_1(drc3)
        drc4 = self.LReLU_drc_4_1(drc4)
        drc4 = self.conv_drc_4_2(drc4)
        drc4 = self.LReLU_drc_4_2(drc4)
        drc4 = self.conv_drc_4_3(drc4)
        drc4 = self.LReLU_drc_4_3(drc4)
        drc4 = tf.concat([drc4, r31, r32, r33], axis=-1)
        drc5 = self.conv_drc_5_1(drc4)
        drc5 = self.LReLU_drc_5_1(drc5)
        drc5 = self.conv_drc_5_2(drc5)
        drc5 = self.LReLU_drc_5_2(drc5)
        drc5 = self.conv_drc_5_3(drc5)
        drc5 = self.LReLU_drc_5_3(drc5)
        drc5 = self.attention(drc5)
        drc6 = self.convtransp_drc_6_1(drc5)
        drc6 = self.LReLU_drc_6_1(drc6)
        drc6 = tf.concat([drc6, c2, d5], axis=-1)
        drc7 = self.conv_drc_7_1(drc6)
        drc7 = self.LReLU_drc_7_1(drc7)
        drc7 = self.conv_drc_7_2(drc7)
        drc7 = self.LReLU_drc_7_2(drc7)
        drc7 = self.conv_drc_7_3(drc7)
        drc7 = self.LReLU_drc_7_3(drc7)
        drc8 = self.conv_out_8(drc7)
        ans_8 = drc8
        drc8 = tf.concat([drc8, drc7], axis=-1)
        drc9 = self.conv_add_1(drc8)
        drc9 = self.LReLU_add_1(drc9)
        drc10 = self.conv_out_16(drc9)
        ans_16 = drc10
        drc10 = tf.concat([drc10, drc9], axis=-1)
        drc11 = self.conv_add_2(drc10)
        drc11 = self.LReLU_add_2(drc11)
        ans_31 = self.conv_out_31(drc11)

        return ans_31


class Discriminator(keras.Model):
    '''
        Discriminator
    '''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = layers.Conv2D(64, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_1 = layers.LeakyReLU(0.2)
        self.conv_2 = layers.Conv2D(128, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_2 = layers.LeakyReLU(0.2)
        self.conv_3 = layers.Conv2D(256, kernel_size=3, strides=(2,2), padding='same')
        self.LReLU_3 = layers.LeakyReLU(0.2)
        self.flatten = layers.Flatten() # flatten layer
        self.fc = layers.Dense(1, activation=tf.nn.sigmoid) # fully connected layer

    def call(self, inputs: list, training):
        x = tf.concat([inputs[0], inputs[1]], axis=-1)
        x = self.conv_1(x)
        x = self.LReLU_1(x)
        x = self.conv_2(x)
        x = self.LReLU_2(x)
        x = self.conv_3(x)
        x = self.LReLU_3(x)
        x = self.flatten(x)
        logits = self.fc(x)
        
        return logits
        

if __name__ == '__main__':
    
    # testing
    generator = Generator()
    discriminator = Discriminator()
    bs = 10
    a = tf.random.normal([bs, 64, 64, 1])
    b = tf.random.normal([bs, 64, 159, 4])
    c = tf.random.normal([bs, 128, 128, 4])
    g_inputs = [a, b, c]
    fake_img = generator(g_inputs)
    print(fake_img.shape)
    d_inputs = [a, fake_img]
    logits = discriminator(d_inputs)
    print(logits.shape)