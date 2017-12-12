from __future__ import division

import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import cv2
from time import gmtime,strftime

from six.moves import xrange

pp = pprint.PrettyPrinter()

get_stddev = lambda x,k_h,k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars,print_info=True)

def imread(path,grayscale = False):
    if(grayscale):
        return scipy.misc.imread(path,flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def get_image(image_path,input_height,input_width,resize_height = 256,resize_width = 256,crop = True,grayscale = False):
    image = imread(image_path,grayscale = grayscale)
    return transform(image,imput_height,input_width,resize_height,resize_width,crop)

def save_images(images,size,image_path):
    return imsave(inverse_transform(images),size,image_path)

def merge_images(images,size):
    h,w = images.shape[1],images.shape[2]
    if(images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h*size[0],w*size[1],c))
        for idx,image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h,i*w:i*w+w,:] = image
        return img
    
    elif images.shape[3] == 1:
        img.np.zeros((h*size[0],w*size[1]))
        for idx,image in enumerate(images):
            i = idx % size[1]
            j = idx // size[2]
            img[j*h:j*h+h,i*w:i*w+w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge (image,size) image paramter must have dimension:H x W or H x W x 3 or HxWx4 ')

def imsave(images,size,path):
    image = np.squeeze(merge(images,size))
    return scipy.misc.imsave(path,image)


def transform(image,input_height,input_width,resize_height = 256,resize_width = 256,crop = True):
    if crop :
        croped_image = center_crop(image,input_height,input_width,resize_height,resize_width)
    else:
        croped_image = scipy.misc.imresize(image,[resize_height,resize_width])
    return np.array(croped_image)/127.5-1

def inverse_transform(images):
    return (images+1.)/2.


def center_crop(image,crop_h,crop_w,resize_h = 256,resize_w = 256):
    if crop_w is None:
        crop_w = crop_h
    h,w = x.shape[:2]
    j = int(round(h-crop_h)/2.)
    i = int(round(w-crop_w)/2.)
    return scipy.misc.imresize(x[j:j+crop_h,i:i+crop_w],[resize_h,resize_w])

def to_json(output_path,*layers):
    # with open(output_path,"w") as layer_f:
    #     lines = ""
    #     for w,b,bn in layers:
    #         layer_idx = w.name.split('/')[0].split('h')[1]
            
    #         B = b.eval()
    pass

def make_gif(images,fname,duration = 2, true_image = False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = image[int(len(images)/duration*t)]
        except:
            x = image[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(makeframe,duration = duration)
    clip.write_gif(fname,fps = len(images)/duration)

def visualize(sess,gan,config,option):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.uniform(-1,1,size = (config.batch_size,gan.z_dim))
        samples = sess.run(gan.sampler,feed_dict = {gan.z:z_sample}) 
        save_images(samples,[image_frame_dim,image_frame_dim],'./samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    elif option == 1:
        values = np.arange(0,1,1./config.batch_size)
        for idx in xrange(gan.z_dim):
            print("[*]%d"% idx)
            z_sample = np.random.uniform(-0.5,0.5,size = (config.batch_size,gan.z_dim))
            for kdx,z in enumerate(z_sample):
                z[idx] = values[kdx]
            
            
            if config.dataset == " ":
                y = np.random.choice(10,config.batch_size)
                y_one_hot = np.zeros((cofig.batch_size,10))
                y_one_hot[np.arange(config.batch_size),y] = 1

                samples = sess.run(gan.sampler,feed_dict = {gan.z:z_sample,gan.y:y_one_hot})
            else:
                samples = sess.run(gan.sampler,feed_dict = {gan.z:z_sample})

            try:
                make_gif(samples,'./samples/test_gif_%s.gif' % (idx))
            except:
                save_image(samples,[image_frame_dim,image_frame_dim],'./samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    elif option == 3:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(gan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, gan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            

            samples = sess.run(gan.sampler, feed_dict={gan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))

    elif option == 4:
        image_set = [] 
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(gan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, gan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]
            image_set.append(sess.run(gan.sampler, feed_dict={gan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))      

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_maniford_size(num_images):
    maniford_h = int(np.floor(np.sqrt(num_images)))
    maniford_w = int(np.floor(np.sqrt(num_images)))
    assert maniford_h*maniford_w == num_images
    return maniford_h,maniford_w