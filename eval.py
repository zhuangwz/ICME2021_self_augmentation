"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from nets import inception_v3

start_time = time.time()
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.flags.DEFINE_string('checkpoint_path_inception_v3',
                       '/ens3_adv_inception_v3.ckpt',
                       'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_dir', 'result/ir2/sim', 'Input directory with images.')
tf.flags.DEFINE_string('initial_dir', '/home/zhuangwz/dataset/ali2019/images1000_val/attack', 'Initial directory with images.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 8, 'How many images process at one time.')
FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    # Run computation
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(
            x_input, num_classes=1001, is_training=False)
    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    with tf.Session() as sess:
        s6.restore(sess, FLAGS.checkpoint_path_inception_v3)
        print(time.time() - start_time)

        sus = 0
        total = 0
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            adv_labels = sess.run(predicted_labels, feed_dict={x_input: images})

            initial_images = np.zeros(batch_shape)
            for i in range(len(filenames)):
                path = os.path.join(FLAGS.initial_dir, filenames[i])
                image = imresize(imread(path, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(
                    np.float) / 255.0
                initial_images[i] = image * 2.0 - 1.0
            ini_labels = sess.run(predicted_labels, feed_dict={x_input: initial_images})

            total += len(filenames)
            sus += np.sum(ini_labels != adv_labels)
            tr = float(sus) / float(total)
            print('transfer rate %f = fool num %d / total %d\n' % (tr, sus, total))


if __name__ == '__main__':
    tf.app.run()
