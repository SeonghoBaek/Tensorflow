import tensorflow as tf
import numpy as np

input_feature_dim = 1024

lstm_sequence_length = 6
lstm_hidden_size = 128
lstm_feature_dim = input_feature_dim
lstm_z_sequence_dim = 128
lstm_linear_transform_input_dim = 2 * lstm_feature_dim

lstm_input = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, lstm_feature_dim])
lstm_linear_transform_weight = tf.Variable(tf.truncated_normal([lstm_linear_transform_input_dim, lstm_z_sequence_dim], stddev=0.1, dtype=tf.float32))
lstm_linear_transform_bias = tf.Variable(tf.zeros([lstm_z_sequence_dim], dtype=tf.float32))

g_encoder_z_local_dim = 128
g_encoder_z_dim = lstm_z_sequence_dim + g_encoder_z_local_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 512
g_encoder_layer2_dim = 256

g_decoder_output_dim = input_feature_dim
g_decoder_layer2_dim = 512
g_decoder_layer1_dim = 768

d_layer_1_dim = 512
d_layer_2_dim = 256

g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, input_feature_dim])
dc_input = tf.placeholder(dtype=tf.float32, shape=[None, input_feature_dim])


def lstm_network(x):
    with tf.varable_scope('lstm'):
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.0)

        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        _, states = tf.nn.dynamic_rnn(lstm_cells, x, dtype=tf.float32)

        states_concat = tf.concat(states[0][1], states[1][1])
        z_sequence_output = tf.add(tf.matmul(states_concat, lstm_linear_transform_weight), lstm_linear_transform_bias)

    return z_sequence_output


def dense(x, n1, n2, name):
    """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param name: name of the entire dense layer.i.e, variable scope name.
        :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def g_encoder_network(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('G_Encoder'):
        g_enc_dense_1 = tf.nn.relu(dense(x, g_encoder_input_dim, g_encoder_layer1_dim, 'g_enc_dense_1'))
        g_enc_dense_2 = tf.nn.relu(dense(g_enc_dense_1, g_encoder_layer1_dim, g_encoder_layer2_dim, 'g_enc_dense_2'))
        # g_enc_z_local = tf.nn.relu(dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_z_local_dim, 'g_enc_z_local'))
        g_enc_z_local = dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_z_local_dim, 'g_enc_z_local')
        return g_enc_z_local


def g_decoder_network(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('G_Decoder'):
        g_dec_dense_2 = tf.nn.relu(dense(x, g_encoder_z_dim, g_decoder_layer2_dim, 'g_dec_dense_2'))
        g_dec_dense_1 = tf.nn.relu(dense(g_dec_dense_2, g_decoder_layer2_dim, g_decoder_layer1_dim, 'g_dec_dense_1'))
        g_dec_output = tf.nn.relu(dense(g_dec_dense_1, g_decoder_layer1_dim, g_decoder_output_dim, 'g_dec_output'))
        return g_dec_output


def r_encoder_network(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('R_Encoder'):
        r_enc_dense1 = tf.nn.relu(dense(x, g_encoder_input_dim, g_decoder_layer1_dim, 'r_enc_dense_1'))
        r_enc_dense2 = tf.nn.relu(dense(r_enc_dense1, g_decoder_layer1_dim, g_decoder_layer2_dim, 'r_enc_dense_2'))
        r_enc_output = dense(r_enc_dense2, g_decoder_layer2_dim, g_encoder_z_dim, 'r_enc_output')
        return r_enc_output


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        dc_den1 = tf.nn.relu(dense(x, input_feature_dim, d_layer_1_dim, name='dc_dense_1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, d_layer_1_dim, d_layer_2_dim, name='dc_dense_2'))
        dc_output = dense(dc_den2, d_layer_2_dim, 1, name='dc_output')
        return dc_den2, dc_output


bn_train = tf.placeholder(tf.bool)


def batch_norm(x):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(bn_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


with tf.variable_scope(tf.get_variable_scope()):
    z_local = g_encoder_network(g_encoder_input)
    z_seq = lstm_network(lstm_input)
    z_enc = tf.concat(z_seq, z_local)
    decoder_output = g_decoder_network(z_enc)

with tf.variable_scope(tf.get_variable_scope()):
    z_renc = r_encoder_network(decoder_output, reuse=True)

with tf.variable_scope(tf.get_variable_scope()):
    feature_real, d_real = discriminator(g_encoder_input)
    feature_fake, d_fake = discriminator(decoder_output, reuse=True)

residual_loss = tf.reduce_mean(tf.abs(decoder_output - g_encoder_input))
feature_matching_loss = tf.reduce_mean(tf.square(feature_real - feature_fake))
conceptual_loss = tf.reduce_mean(tf.square(z_enc - z_renc))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
discriminator_loss = d_loss_real + d_loss_fake
