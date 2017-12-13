from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import re
from six.moves import xrange
import time
import math
from glob import glob

from ops import *
from utils import *


import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



def conv_out_size_same(size,stride):
    '''
        return the conv2d ouput size
    '''
    return int(math.ceil(float(size)/float(stride)))


class GAN(object):
    def __init__(self,sess,input_height = 256,input_width = 256,crop = True,batch_size = 8,sample_num = 64,
        output_height = 256,output_width = 256,z_dim = 100,gf_dim = 64,df_dim = 64,gfc_dim = 1024,
        dfc_dim = 1024,c_dim = 3,dataset_dim = 'default',input_fname_pattern = '*.jpg',checkpoint_dir = None,sample_dir = None):
        
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = gf_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = gfc_dim
 
        self.d_bn1 = batch_norm(name = 'd_bn1')
        self.d_bn2 = batch_norm(name = 'd_bn2')

        
        self.g_bn0 = batch_norm(name = 'g_bn0')
        self.g_bn1 = batch_norm(name = 'g_bn1')
        self.g_bn2 = batch_norm(name = 'g_bn2')
        self.g_bn3 = batch_norm(name = 'g_bn3')
        self.g_bn4 = batch_norm(name = 'g_bn4')
        self.g_bn5 = batch_norm(name = 'g_bn5')
        self.g_bn6 = batch_norm(name = 'g_bn6')
        

        self.g_y_bn0 = batch_norm(name = 'g_y_bn0')
        self.g_y_bn1 = batch_norm(name = 'g_y_bn1')
        self.g_y_bn2 = batch_norm(name = 'g_y_bn2')
        self.g_y_bn3 = batch_norm(name = 'g_y_bn3')

        self.g_z_bn0 = batch_norm(name = 'g_z_bn0')
        self.g_z_bn1 = batch_norm(name = 'g_z_bn1')
        self.g_z_bn2 = batch_norm(name = 'g_z_bn2')
        self.g_z_bn3 = batch_norm(name = 'g_z_bn3')



        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir


        
        self.data = glob(os.path.join("/data",self.dataset_name,self.input_fname_pattern))
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >=3:
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)
        self.build_model()


    def build_model(self):
        '''
            y : projection images [bz,512,512,3]
            inputs: real images

        '''
        self.y = tf.palcceholder(tf.float32,[self.batch_size,self.input_height,self.input_width,self.c_dim],name = 'y')

        if self.crop:
            image_dims = [self.output_height,self.output_width,self.c_dim]
        else:
            image_dims = [self.input_height,self.input_width,self.c_dim]


        self.inputs = tf.placeholder(tf.float32,[self.batch_size,self.input_height,self.input_width,self.c_dim],name='real_images')
        
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32,[self.batch_size,self.z_dim],name = 'z')
        self.z_sum = histogram_summary("z",self.z)

        self.G = self.generator(self.z,self.y)
        self.D,self.D_logits = self.discriminator(inputs,self.y,reuse = False)
        #self.sampler = self.sampler(self.z,self.y)
        self.D_,self.D_logits_ = self.discriminator(self.G,self.y,reuse = True)

        self.d_sum = histogram_summary("d",self.D)
        self.d__sum = histogram_summary("d_",self.D_)
        self.G_sum = image_summary("G",self.G)

        def sigmoid_cross_entropy_with_logits(logits,labels):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,labels = labels)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,targets = labels)
            
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits,tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_,tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_,tf.ones_like(self.D_)))


        self.d_loss_real_sum = scalar_summary("d_loss_real",self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake",self.d_loss_fake)

        self.d_loss = self.d_loss_fake + self.d_loss_real

        self.g_loss_sum = scalar_summary("g_loss",self.g_loss)
        self.d_loss_sum = scalar_summary('d_loss',self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self,config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.d_loss,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum,self.d__sum,self.G_sum,self.d_loss_fake_sum,self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs",self.sess.graph)


        sample_z = np.random(-1,1,size = (self.sample_num,self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [get_image(sample_file,self.input_height,self.input_width,512,512,self.crop,self.grayscale) for sample_file in sample_files]

        if self.grayscale:
            sample_inputs = np.array(sample).astype(np.float32)[:,:,:,None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load,checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print("[*] load SUCCESS")
        else:
            print("[!] load failed")

        self.data_y = glob(os.path.join("/data/projection",config.dataset,self.input_fname_pattern))
        self.data_r = glob(os.path.join("/data/real",config.dataset,self.input_fname_pattern))

        if len(self.data_y)!=len(self.data_r):
                raise Exception("dataset error")

        for epoch in xrange(config.epoch):
            batch_idx = min(len(self.data_y),config.train_size)
            #self.data = glob(os.path.join("/data",config.dataset,self.input_fname_pattern))
            #batch_idx = min(len(self.data),config.train_size)

            for idx in xrange(0,batch_idx):
                batch_y_files = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_r_files = self.data_r[idx*config.batch_size:(idx+1)*config.batch_size]
                #batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file,self.input_height,self.input_width,512,512,False,False) for batch_file in batch_files]
                batch_y = [get_image(batch_y_file,self.input_height,self.input_width,512,512,False,False) for batch_y_file in batch_y_files]
                batch_r = [get_image(batch_r_file,self.input_height,self.input_width,512,512,False,False) for batch_r_file in batch_y_files]
                # if self.grayscale:
                #     batch_images = np.array(batch).astype(np.float32)[:,:,:,None] 
                # else:
                #     batch_images = np.array(batch).astype(np.float32)   
                batch_y_images = np.array(batch_y).astype(np.float32)
                batch_r_images = np.array(batch_r).astype(np.float32)

                batch_z = np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)
            

                _,summary_str = self.sess.run([d_optim,self.d_sum],feed_dict = {self.inputs:batch_r_images,self.y:batch_y_images,self.z:batch_z})##??有问题

                self.writer.add_summary(summary_str,counter)

                _,summary_str = self.sess.run([g_optim,self.g_sum],feed_dict = {self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})

                _,summary_str = self.sess.run([g_optim,self.g_sum],feed_dict = {self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})

                errD_fake = self.d_loss_fake.eval({self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})#???
                errD_real = self.d_loss_real.eval({self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})
                errG = self.g_loss({self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})
            counter += 1

            print("Epoch : [%2d],[%4d/%4d] time: %4.4f, d_loss : %.8f,g_loss = %.8f " (epoch,idx,batch_idx,time.time()-start_time,errD_fake+errD_real,errG))

            if np.mod(counter,100) == 1:
                try:
                    samples,d_loss,g_loss = self.sess.run([self.sampler,self.d_loss,self.g_loss],feed_dict = {self.z:batch_z,self.y:batch_y_images,self.inputs:batch_r_images})
                    save_images(samples,image_maniford_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir,epoch,idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                except:
                    print("one pic error!...")
            
            if np.mod(counter,500) == 2:
                self.save(config.checkpoint_dir, counter)
        ############################################################################  
        pass


    def sampler(self,z,y):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            
            self.y0 = conv2d(y,output_dim = 32,name = 'g_y_h0')#batch_size*512*512*3->batch_size*256*256*32
            y0 = tf.nn.relu(self.g_y_bn0(self.y0))
            self.y1 = conv2d(y0,output_dim = 64,name = 'g_y_h1')#batch_size*256*256*32->128*128*64
            y1 = tf.nn.relu(self.g_y_bn1(self.y1))
            self.y2 = conv2d(y1,output_dim = 128,name = 'g_y_h2')#bz*128*128*64-> bz*64*64*128
            y2 = tf.nn.relu(self.g_y_bn2(self.y2))
            
            self.z_ = linear(z,64*16*16,name = 'g_z_h0')
            self.z0 = tf.reshape(self.z_,[-1,16,16,64])
            z0 = tf.nn.relu(self.g_z_bn0(self.z0))
            self.z1 = deconv2d(z0,[self.batch_size,32,32,64],name = 'g_z_h1')
            z1 = tf.nn.relu(self.g_z_bn1(self.z1))
            self.z2 = deconv2d(z1,[self.batch_size,64,64,64],name = 'g_z_h2')
            z2 = tf.nn.relu(self.g_z_bn2(self.z2))
            
            self.z_con_y = tf.concat([z2,y2],3) #batch_size*64*64*(128+64)
            self.z_con_y_0 = conv2d(self.z_con_y,256,3,3,1,1,name = 'g_con_h0') #kernel size = 3*3, strides = 1 ->[bz,64,64,256]
            z_con_y_0 = tf.nn.relu(self.g_bn0(self.z_con_y_0)) 
            self.z_con_y_1 = conv2d(z_con_y_0,512,3,3,2,2,name = 'g_con_h1') #kernel = [1,3,3,1] strides = [1,2,2,1] ->[bz,,32,32,512]
            z_con_y_1 = tf.nn.relu(self.g_bn1(self.z_con_y_1))
            self.z_con_y_2 = deconv2d(z_con_y_1,[self.batch_size,64,64,256],3,3,name = 'g_con_h2')#->[bz,64,64,256]
            z_con_y_2 = tf.nn.relu(self.g_bn2(self.z_con_y_2))
            self.z_con_y_3 = deconv2d(z_con_y_2,[self.batch_size,128,128,128],3,3,name ='g_con_h3')
            z_con_y_3 = tf.nn.relu(self.g_bn3(self.z_con_y_3))
            self.z_con_y_4 = deconv2d(z_con_y_3,[self.batch_size,256,256,32],4,4,name ='g_con_h4')
            z_con_y_4 = tf.nn.relu(self.g_bn4(self.z_con_y_4))
            self.z_con_y_5 = deconv2d(z_con_y_4,[self.batch_size,512,512,3],5,5,name ='g_con_h4')
            z_con_y_5 = tf.nn.relu(self.g_bn5(self.z_con_y_5))

            return z_con_y_5



    # def discriminator(self,image,y,reuse = False):
    #     with tf.variable_scope('descriminator') as scope:
    #         if reuse:
    #             scope.reuse_variables()

    #         yb= tf.reshape(y,[batch_size,512,512,3])
    #         x = tf.concat([image,yb],3)
    #         h0 = lrelu(conv2d(x,self.c_dim+self.y_dim,name = 'd_h0_conv'))
    #         h1 = lrelu(self.d_bn1(conv2d(h0,self.df_dim+self.y_dim,name = 'd_h1_conv')))
    #         h1 = tf.reshape(h1,[self.batch_size,-1])
    #         h2 = lrelu(self.d_bn2(linear(h1,self.dfc_dim,'d_h2_lin')))
    #         h3 = linear(h2,1,'d_h3_lin')
    #         return tf.nn.sigmoid(h3)    

    def discriminator(self,images,projection,reuse = False):
        '''
            images: real or generated images [batch_size,512,512,3]
            projction: projection images [batch_size,512,512,3]
        '''
        with tf.variable_scope("descriminator") as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.concat([images,projection],3)
            h0 = lrelu(con2d(x,32,name = 'd_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0,64,name = 'd_h1_conv')))
            h1 = tf.reshape(h1,[self.batch_size,-1])
            h2 = lrelu(self.d_bn2(linear(h1,self.dfc_dim,'d_h2_lin')))
            h3 = linear(h2,1,'d_h_lin')
            return tf.nn.sigmoid(h3),h3
        

    def generator(self,z,y):
        '''
            z uniform_random_variables
            y projection images batch*512*512*3
        '''
        with tf.variable_scope('generator') as scope:
            #self.y_,self.yh0_w,self.yh0_b = conv2d(y,output_dim = 32,name = 'g_yh0') 
            self.y0 = conv2d(y,output_dim = 32,name = 'g_y_h0')#batch_size*512*512*3->batch_size*256*256*32
            y0 = tf.nn.relu(self.g_y_bn0(self.y0))
            self.y1 = conv2d(y0,output_dim = 64,name = 'g_y_h1')#batch_size*256*256*32->128*128*64
            y1 = tf.nn.relu(self.g_y_bn1(self.y1))
            self.y2 = conv2d(y1,output_dim = 128,name = 'g_y_h2')#bz*128*128*64-> bz*64*64*128
            y2 = tf.nn.relu(self.g_y_bn2(self.y2))
            
            self.z_ = linear(z,64*16*16,name = 'g_z_h0')
            self.z0 = tf.reshape(self.z_,[-1,16,16,64])
            z0 = tf.nn.relu(self.g_z_bn0(self.z0))
            self.z1 = deconv2d(z0,[self.batch_size,32,32,64],name = 'g_z_h1')
            z1 = tf.nn.relu(self.g_z_bn1(self.z1))
            self.z2 = deconv2d(z1,[self.batch_size,64,64,64],name = 'g_z_h2')
            z2 = tf.nn.relu(self.g_z_bn2(self.z2))
            
            self.z_con_y = tf.concat([z2,y2],3) #batch_size*64*64*(128+64)
            self.z_con_y_0 = conv2d(self.z_con_y,256,3,3,1,1,name = 'g_con_h0') #kernel size = 3*3, strides = 1 ->[bz,64,64,256]
            z_con_y_0 = tf.nn.relu(self.g_bn0(self.z_con_y_0)) 
            self.z_con_y_1 = conv2d(z_con_y_0,512,3,3,2,2,name = 'g_con_h1') #kernel = [1,3,3,1] strides = [1,2,2,1] ->[bz,,32,32,512]
            z_con_y_1 = tf.nn.relu(self.g_bn1(self.z_con_y_1))
            self.z_con_y_2 = deconv2d(z_con_y_1,[self.batch_size,64,64,256],3,3,name = 'g_con_h2')#->[bz,64,64,256]
            z_con_y_2 = tf.nn.relu(self.g_bn2(self.z_con_y_2))
            self.z_con_y_3 = deconv2d(z_con_y_2,[self.batch_size,128,128,128],3,3,name ='g_con_h3')
            z_con_y_3 = tf.nn.relu(self.g_bn3(self.z_con_y_3))
            self.z_con_y_4 = deconv2d(z_con_y_3,[self.batch_size,256,256,32],4,4,name ='g_con_h4')
            z_con_y_4 = tf.nn.relu(self.g_bn4(self.z_con_y_4))
            self.z_con_y_5 = deconv2d(z_con_y_4,[self.batch_size,512,512,3],5,5,name ='g_con_h4')
            z_con_y_5 = tf.nn.relu(self.g_bn5(self.z_con_y_5))

            return z_con_y_5

    # def sampler(self,z,y = None):
    #     with tf.variable('sampler') as scope:
    #         scope.reuse_variables()

    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name,self.batch_size,self.output_height,self.output_width)

    
    def save(self,checkpoint_dir,step):
        model_name = "myGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir,self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess,os.path.join(checkpoint_dir,model_name),global_step=step)


    def load(self,  checkpoint_dir):
        import re
        print ("[*] reading checkpoints")
        checkpoint_dir = os.path.join(checkpoint_dir,self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and  ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(checkpoint_dir,ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print ("success to read {}".format(ckpt_name))
            return True,counter

        else:
            print ("[*] failed to load checkpoint ")
            return False,0
         
        