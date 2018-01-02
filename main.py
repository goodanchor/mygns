import os
import scipy.misc
import numpy as np


from Gan_model import GAN

from utils import pp,visualize,to_json,show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch",60,"Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [64]")
flags.DEFINE_integer("input_height",512, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 512, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 512, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 512, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    # if FLAGS.input_width is None:
    #     FLAGS.input_width = FLAGS.input_height
    # if FLAGS.output_width is None:
    #     FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # run_config = tf.ConfigProto()
    # # run_config.gpu_options
    # run_config.gpu_options.allow_growth = True
    
    #with tf.Session(config=run_config) as sess:
    with tf.Session() as sess:
        gan = GAN(sess,
                input_height = FLAGS.input_height,
                input_width = FLAGS.input_width,
                crop = FLAGS.crop,
                batch_size = FLAGS.batch_size,
                sample_num = FLAGS.batch_size,
                output_height = FLAGS.input_height,
                output_width = FLAGS.input_width,
                z_dim = 100,
                input_fname_pattern =FLAGS.input_fname_pattern,
                checkpoint_dir = FLAGS.checkpoint_dir,
                sample_dir = FLAGS.sample_dir)
        show_all_variables()

    if FLAGS.train:
        gan.train(FLAGS)
    else:
        if not gan.load(FLAGS.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first ten run test mode")

    OPTION = 1
    #visualize(sess,gan,FLAGS,OPTION)

if __name__ == "__main__":
    tf.app.run()