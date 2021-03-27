from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.stats as st
from scipy.misc import imread, imsave
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')
tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')
tf.flags.DEFINE_integer('gpu_num', 2, 'number of used gpu.')
tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')
tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')
tf.flags.DEFINE_boolean('use_input', True, 'use input or not.')
tf.flags.DEFINE_boolean('use_conv', True, 'use conv or not.')
tf.flags.DEFINE_integer('sample_num', 4, 'the size of gradient.')
tf.flags.DEFINE_float('sample_variance', 0.12, 'the size of gradient.')
tf.flags.DEFINE_string('checkpoint_path',
                       '/ckpt_models/',
                       'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('input_dir',
                       '/ImageNet_dir/',
                       'Input directory with images.')
tf.flags.DEFINE_string('output_dir',
                       '/Output_dir/',
                       'Output directory with images.')
FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152.ckpt'),
    'densenet': os.path.join(FLAGS.checkpoint_path, 'tf-densenet161.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel_9 = gkern(9, 3).astype(np.float32)
stack_kernel_9 = np.stack([kernel_9, kernel_9, kernel_9]).swapaxes(2, 0)
stack_kernel_9 = np.expand_dims(stack_kernel_9, 3)

kernel_11 = gkern(11, 3).astype(np.float32)
stack_kernel_11 = np.stack([kernel_11, kernel_11, kernel_11]).swapaxes(2, 0)
stack_kernel_11 = np.expand_dims(stack_kernel_11, 3)

kernel_15 = gkern(15, 3).astype(np.float32)
stack_kernel_15 = np.stack([kernel_15, kernel_15, kernel_15]).swapaxes(2, 0)
stack_kernel_15 = np.expand_dims(stack_kernel_15, 3)


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
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
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


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def inceptionv3_model(x):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v3, end_points_v3


def inceptionv4_model(x):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v4, end_points_v4


def inceptionresnetv2_model(x):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_res_v2, end_points_res_v2


def resnet152_model(x):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet


def average_grads(grads_tower):
    average_grads = []
    for grad_and_vars in zip(*grads_tower):
        grads = []
        for g, _ in grad_and_vars:
            grads.append(g)
        grad = tf.reduce_mean(grads, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)
    return average_grads


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2 * FLAGS.max_epsilon / 255.0

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        num_per_gpu = FLAGS.sample_num // FLAGS.gpu_num
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        num_iter = FLAGS.num_iter
        alpha = eps / num_iter
        momentum = FLAGS.momentum
        num_classes = 1001
        batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
        x_initial = tf.placeholder(tf.float32, shape=batch_shape)
        adv_tf = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_initial + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_initial - eps, -1.0, 1.0)
        adv_clip = tf.clip_by_value(adv_tf, x_min, x_max)

        noise_tf = tf.placeholder(dtype='float32', shape=[None, FLAGS.image_height, FLAGS.image_width, 3])
        noise_np = tf.nn.depthwise_conv2d(noise_tf, stack_kernel_15, strides=[1, 1, 1, 1], padding='SAME')
        noise_np = noise_np / tf.reduce_mean(tf.abs(noise_np), [1, 2, 3], keep_dims=True)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_initial, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y_label = tf.placeholder(tf.int64, shape=[FLAGS.sample_num, num_classes])
        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        one_hot_single = tf.one_hot(predicted_labels, num_classes)
        one_hot = tf.tile(one_hot_single, [FLAGS.sample_num, 1])

        grad = tf.placeholder(tf.float32, shape=batch_shape)

        x_dev = x_input + momentum * alpha * grad

        d_shape = [FLAGS.sample_num, FLAGS.image_height, FLAGS.image_width, 3]
        d = tf.random_normal(shape=d_shape)

        # deviation augmentation
        if FLAGS.use_input:
            x_dev = x_dev + FLAGS.sample_variance * tf.sign(d)

        # multi-gpu operation
        grads_tower = None
        for i in range(FLAGS.gpu_num):
            with tf.device('/gpu:%d' % i):
                print('loading on gpu %d of %d' % (i + 1, FLAGS.gpu_num))
                x_dev_gpu = x_dev[i * num_per_gpu: (i + 1) * num_per_gpu]
                y_gpu = y_label[i * num_per_gpu: (i + 1) * num_per_gpu]

                logits_v3, end_points_v3 = inceptionv3_model(x_dev_gpu)
                logits_v4, end_points_v4 = inceptionv4_model(x_dev_gpu)
                logits_v2, end_points_v2 = inceptionresnetv2_model(x_dev_gpu)
                logits_152, end_points_152 = resnet152_model(x_dev_gpu)
                logits = (logits_v3 + logits_v4 + logits_v2 + logits_152) / 4
                cross_entropy = tf.losses.softmax_cross_entropy(y_gpu, logits)

                # SI-TIM
                x_nes_2 = 1 / 2 * x_dev_gpu
                logits_v3_2, end_points_v3 = inceptionv3_model(x_nes_2)
                logits_v4_2, end_points_v4 = inceptionv4_model(x_nes_2)
                logits_v2_2, end_points_v2 = inceptionresnetv2_model(x_nes_2)
                logits_152_2, end_points_152 = resnet152_model(x_nes_2)
                logits_2 = (logits_v3_2 + logits_v4_2 + logits_v2_2 + logits_152_2) / 4
                cross_entropy_2 = tf.losses.softmax_cross_entropy(y_gpu, logits_2)

                x_nes_4 = 1 / 4 * x_dev_gpu
                logits_v3_4, end_points_v3 = inceptionv3_model(x_nes_4)
                logits_v4_4, end_points_v4 = inceptionv4_model(x_nes_4)
                logits_v2_4, end_points_v2 = inceptionresnetv2_model(x_nes_4)
                logits_152_4, end_points_152 = resnet152_model(x_nes_4)
                logits_4 = (logits_v3_4 + logits_v4_4 + logits_v2_4 + logits_152_4) / 4
                cross_entropy_4 = tf.losses.softmax_cross_entropy(y_gpu, logits_4)

                x_nes_8 = 1 / 8 * x_dev_gpu
                logits_v3_8, end_points_v3 = inceptionv3_model(x_nes_8)
                logits_v4_8, end_points_v4 = inceptionv4_model(x_nes_8)
                logits_v2_8, end_points_v2 = inceptionresnetv2_model(x_nes_8)
                logits_152_8, end_points_152 = resnet152_model(x_nes_8)
                logits_8 = (logits_v3_8 + logits_v4_8 + logits_v2_8 + logits_152_8) / 4
                cross_entropy_8 = tf.losses.softmax_cross_entropy(y_gpu, logits_8)

                cross_entropy = cross_entropy + cross_entropy_2 + cross_entropy_4 + cross_entropy_8

                if FLAGS.use_conv:
                    x_conv_9 = tf.nn.depthwise_conv2d(x_dev_gpu, stack_kernel_9, strides=[1, 1, 1, 1], padding='SAME')
                    logits_v3_9, _ = inceptionv3_model(x_conv_9)
                    logits_v4_9, _ = inceptionv4_model(x_conv_9)
                    logits_v2_9, _ = inceptionresnetv2_model(x_conv_9)
                    logits_152_9, _ = resnet152_model(x_conv_9)
                    logits_9 = (logits_v3_9 + logits_v4_9 + logits_v2_9 + logits_152_9) / 4
                    cross_entropy_9 = tf.losses.softmax_cross_entropy(y_gpu, logits_9)

                    x_conv_11 = tf.nn.depthwise_conv2d(x_dev_gpu, stack_kernel_11, strides=[1, 1, 1, 1], padding='SAME')
                    logits_v3_11, _ = inceptionv3_model(x_conv_11)
                    logits_v4_11, _ = inceptionv4_model(x_conv_11)
                    logits_v2_11, _ = inceptionresnetv2_model(x_conv_11)
                    logits_152_11, _ = resnet152_model(x_conv_11)
                    logits_11 = (logits_v3_11 + logits_v4_11 + logits_v2_11 + logits_152_11) / 4
                    cross_entropy_11 = tf.losses.softmax_cross_entropy(y_gpu, logits_11)

                    cross_entropy = cross_entropy + cross_entropy_9 + cross_entropy_11

                noise = tf.gradients(cross_entropy, x_dev_gpu)
                grads_tower = tf.concat([grads_tower, noise], axis=0) if grads_tower else noise

        grads = tf.expand_dims(tf.reduce_mean(tf.reduce_mean(grads_tower,axis=0),axis=0),axis=0)
        grads = tf.nn.depthwise_conv2d(grads, stack_kernel_11, strides=[1, 1, 1, 1], padding='SAME')
        grads = grads / tf.reduce_mean(tf.abs(grads), [1, 2, 3], keep_dims=True)

        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        s1.restore(sess, model_checkpoint_map['inception_v3'])
        s2.restore(sess, model_checkpoint_map['inception_v4'])
        s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
        s4.restore(sess, model_checkpoint_map['resnet_v2'])

        idx = 0
        l2_diff = 0
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            idx = idx + 1
            print("start the i={} attack".format(idx))

            one_hot_np = sess.run(one_hot, feed_dict={x_initial: images})
            grad_current = np.zeros(shape=batch_shape)
            advs = images
            for i in range(FLAGS.num_iter):
                grad_np = sess.run(grads, feed_dict={x_input: advs, y_label: one_hot_np, grad: grad_current})
                grad_current = grad_np + momentum * grad_current
                advs = advs + alpha * np.sign(grad_current)
                advs = sess.run(adv_clip, feed_dict={x_initial: images, adv_tf: advs})
                grads_tower = None

            adv_images = advs
            save_images(adv_images, filenames, os.path.join(FLAGS.output_dir))
            diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
            l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

        print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
