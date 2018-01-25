import tensorflow as tf

from layers import *


def encoder(input, phase_train):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    # conv(input, name, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu)
    conv1 = conv(input, name='conv1', filter_dims=[3, 3, 1], stride_dims=[2, 2], padding='SAME', non_linear_fn=tf.nn.relu, bias=False)

    conv1 = batch_norm(conv1, phase_train=phase_train)

    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    conv2 = conv(conv1, name='conv2', filter_dims=[3, 3, 32], stride_dims=[2, 2], padding='SAME', non_linear_fn=tf.nn.relu, bias=False)

    conv2 = batch_norm(conv2, phase_train=phase_train)

    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    conv3 = conv(conv2, name='conv3', filter_dims=[3, 3, 16], stride_dims=[2, 2], padding='SAME', non_linear_fn=tf.nn.relu, bias=False)

    conv3 = batch_norm(conv3, phase_train=phase_train)

    # FC: output_dim: 100, no non-linearity
    # fc(input, name, out_dim, non_linear_fn=tf.nn.relu)
    fc1 = fc(conv3, name='fc1', out_dim=128, non_linear_fn=None)

    return fc1
    raise NotImplementedError


def decoder(input, keep_prop):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    input = tf.nn.dropout(input, keep_prob=keep_prop)
    fc2 = fc(input, name='fc2', out_dim=256, non_linear_fn=tf.nn.relu)

    # Reshape to [batch_size, 4, 4, 8]
    feature = tf.reshape(fc2, shape=[-1, 4, 4, 16])

    print('reshape', feature.get_shape().as_list())

    # deconv(input, name, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu)

    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    # deconv1 = deconv(feature, name='deconv1', filter_dims=[3, 3, 8], stride_dims=[2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    deconv1 = deconv_v2(feature, name='deconv1', filter_dims=[3, 3, 8], output_dims=[7, 7, 8], padding='SAME', non_linear_fn=tf.nn.relu)
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    # deconv2 = deconv(deconv1, name='deconv2', filter_dims=[8, 8, 1], stride_dims=[2, 2], padding='VALID', non_linear_fn=tf.nn.relu)
    deconv2 = deconv_v2(deconv1, name='deconv2', filter_dims=[3, 3, 16], output_dims=[14, 14, 16], padding='SAME',
                        non_linear_fn=tf.nn.relu)
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    # deconv3 = deconv(deconv2, name='deconv3', filter_dims=[7, 7, 1], stride_dims=[1, 1], padding='VALID', non_linear_fn=tf.nn.sigmoid)
    deconv3 = deconv_v2(deconv2, name='deconv3', filter_dims=[3, 3, 1], output_dims=[28, 28, 1], padding='SAME',
                        non_linear_fn=tf.nn.sigmoid)

    #deconv3 = deconv(feature, name='deconv1', filter_dims=[3, 3, 1], stride_dims=[7, 7], padding='SAME', non_linear_fn=tf.nn.sigmoid)

    return deconv3

    raise NotImplementedError


def autoencoder(input_shape):
    assert(len(input_shape) == 4)
    # Define place holder with input shape
    input_image = tf.placeholder(dtype=tf.float32, shape=input_shape)
    phase_train = tf.placeholder(dtype=tf.bool)
    keep_prop = tf.placeholder(dtype=tf.float32)

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(input_image, phase_train)
        # Pass encoding into decoder to obtain reconstructed image
        reconstructed_image = decoder(encoding, keep_prop)
        # Return input image (placeholder) and reconstructed image
        return input_image, reconstructed_image, phase_train, keep_prop
        pass
