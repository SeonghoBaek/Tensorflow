import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


sess = tf.Session()

data_dir = 'temp'
mnist = read_data_sets(data_dir)

train_xdata = np.array([np.reshape(x, [28, 28]) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, [28, 28]) for x in mnist.test.images])
train_label = mnist.train.labels
test_label = mnist.test.labels

batch_size = 128
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_label) + 1
num_channels = 1
generations = 100
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = [batch_size, image_height, image_width, num_channels]
x_input = tf.placeholder(dtype=tf.float32, shape=x_input_shape)
y_target = tf.placeholder(dtype=tf.int32, shape=[batch_size])
eval_input_shape = [batch_size, image_height, image_width, num_channels]
eval_input = tf.placeholder(dtype=tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(dtype=tf.int32, shape=[batch_size])
phase_train = tf.placeholder(tf.bool)
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
conv1_beta = tf.Variable(tf.ones([conv1_features], dtype = tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))
conv2_beta = tf.Variable(tf.ones([conv2_features], dtype=tf.float32))

resulting_width = image_width//(max_pool_size1 * max_pool_size2)
resulting_height = image_height//(max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height * conv2_features
fused_input_size = 7350  # test

#full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
full1_weight = tf.Variable(tf.truncated_normal([fused_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))

full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
full1_beta = tf.Variable(tf.ones([fully_connected_size1], dtype=tf.float32))


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
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def my_bn_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    #logit1 = tf.nn.bias_add(conv1, conv1_bias)
    #swish1 = tf.nn.relu(logit1)

    conv1 = batch_norm(conv1)

    swish1 = tf.multiply(2*conv1, tf.nn.sigmoid(conv1_beta * conv1))
    #swish1 = tf.multiply(logit1, tf.nn.sigmoid(logit1))
    max_pool1 = tf.nn.max_pool(swish1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
    feature1_shape = max_pool1.get_shape().as_list()
    feature1 = tf.reshape(max_pool1, [feature1_shape[0], -1])

    print('Feature1 ' + str(feature1.get_shape().as_list()))

    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm(conv2)

    #logit2 = tf.nn.bias_add(conv2, conv2_bias)
    #swish2 = tf.nn.relu(logit2)
    swish2 = tf.multiply(2*conv2, tf.nn.sigmoid(conv2_beta * conv2))
    #swish2 = tf.multiply(logit2, tf.nn.sigmoid(logit2))
    max_pool2 = tf.nn.max_pool(swish2, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
    feature2_shape = max_pool2.get_shape().as_list()
    feature2 = tf.reshape(max_pool2, [feature2_shape[0], -1])
    print('Feature2 ' + str(feature2.get_shape().as_list()))

    fused_feature = tf.concat([feature1, feature2], axis=1)

    print('Fused Feature ' + str(fused_feature.get_shape().as_list()))

    #final_conv_shape = max_pool2.get_shape().as_list()
    #print(final_conv_shape)
    #final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    #flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    #logit3 = tf.add(tf.matmul(flat_output, full1_weight), full1_bias)
    logit3 = tf.add(tf.matmul(fused_feature, full1_weight), full1_bias)
    #fully_connected1 = tf.nn.relu(logit3)
    fully_connected1 = tf.multiply(2*logit3, tf.nn.sigmoid(full1_beta * logit3))
    #fully_connected1 = tf.multiply(logit3, tf.nn.sigmoid(logit3))

    print(fully_connected1.get_shape().as_list())

    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return final_model_output


def my_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    logit1 = tf.nn.bias_add(conv1, conv1_bias)
    #swish1 = tf.nn.relu(logit1)

    swish1 = tf.multiply(2*logit1, tf.nn.sigmoid(conv1_beta * logit1))
    #swish1 = tf.multiply(logit1, tf.nn.sigmoid(logit1))
    max_pool1 = tf.nn.max_pool(swish1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
    feature1_shape = max_pool1.get_shape().as_list()
    feature1 = tf.reshape(max_pool1, [feature1_shape[0], -1])

    print('Feature1 ' + str(feature1.get_shape().as_list()))

    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    logit2 = tf.nn.bias_add(conv2, conv2_bias)
    #swish2 = tf.nn.relu(logit2)
    swish2 = tf.multiply(2*logit2, tf.nn.sigmoid(conv2_beta * logit2))
    #swish2 = tf.multiply(logit2, tf.nn.sigmoid(logit2))
    max_pool2 = tf.nn.max_pool(swish2, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
    feature2_shape = max_pool2.get_shape().as_list()
    feature2 = tf.reshape(max_pool2, [feature2_shape[0], -1])
    print('Feature2 ' + str(feature2.get_shape().as_list()))

    fused_feature = tf.concat([feature1, feature2], axis=1)

    print('Fused Feature ' + str(fused_feature.get_shape().as_list()))

    #final_conv_shape = max_pool2.get_shape().as_list()
    #print(final_conv_shape)
    #final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    #flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    #logit3 = tf.add(tf.matmul(flat_output, full1_weight), full1_bias)
    logit3 = tf.add(tf.matmul(fused_feature, full1_weight), full1_bias)
    #fully_connected1 = tf.nn.relu(logit3)
    fully_connected1 = tf.multiply(2*logit3, tf.nn.sigmoid(full1_beta * logit3))
    #fully_connected1 = tf.multiply(logit3, tf.nn.sigmoid(logit3))

    print(fully_connected1.get_shape().as_list())

    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return final_model_output


#model_output = my_conv_net(x_input)
model_output = my_bn_conv_net(x_input)
#test_model_output = my_conv_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.nn.softmax(model_output)
#test_prediction = tf.nn.softmax(test_model_output)


def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))

    return 100.*num_correct/batch_predictions.shape[0]


my_optimizer = tf.train.AdamOptimizer(0.01)
train_step = my_optimizer.minimize(loss)

saver = tf.train.Saver()

saver.restore(sess, '/Users/major/model.ckpt')

#init = tf.global_variables_initializer()
#sess.run(init)

for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_label[rand_index]
    train_dict = {x_input:rand_x, y_target:rand_y, phase_train: True}

    _, l, p, _, _, _ = sess.run([train_step, loss, prediction, conv1_beta, conv2_beta, full1_beta], feed_dict=train_dict)

    if (i+1) % eval_every == 0:
        print(l, get_accuracy(p, rand_y))



saver.save(sess, '/Users/major/model.ckpt')