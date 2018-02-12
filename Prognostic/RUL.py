import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale


def windows(nrows, size):
    start, step = 0, 2
    while start < nrows:
        yield start, start + size
        start += step


def segment_signal(features, labels, window_size = 15):
    segments = np.empty((0, window_size))
    segment_labels = np.empty((0))
    nrows = len(features)

    print('Num row: ', nrows)

    for (start, end) in windows(nrows, window_size):
        if len(data.iloc[start:end]) == window_size:
            segment = features[start:end].T  #Transpose to get segment of size 24 x 15
            label = labels[(end-1)]
            segments = np.vstack([segments, segment])
            segment_labels = np.append(segment_labels, label)

    segments = segments.reshape(-1,24,window_size,1) # number of features  = 24
    segment_labels = segment_labels.reshape(-1,1)
    return segments, segment_labels


# id,cycle,setting1,setting2,setting3,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,RUL
print('Read CSV')
data = pd.read_csv("PHM08.csv")
print('Scale CSV')
features = scale(data.iloc[:, 2:26])  # select required columns and scale them
labels = data.iloc[:, 26]  # select RUL

print('Segmentation')
segments, labels = segment_signal(features, labels)

print('Prepare data')
train_test_split = np.random.rand(len(segments)) < 0.70
train_x = segments[train_test_split]
train_y = labels[train_test_split]
test_x = segments[~train_test_split]
test_y = labels[~train_test_split]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def apply_conv(x, kernel_height, kernel_width, num_channels, depth):
    weights = weight_variable([kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding="VALID"), biases))


def apply_max_pool(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 1, stride_size, 1], padding="VALID")


num_labels = 1
batch_size = 10
num_hidden = 800
learning_rate = 0.0001
training_epochs = 30
input_height = 24
input_width = 15
num_channels = 1
total_batches = train_x.shape[0] // batch_size

X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

print('Build Graph')
c = apply_conv(X, kernel_height=24, kernel_width=4, num_channels=1, depth=8)
print(c.get_shape().as_list())
p = apply_max_pool(c, kernel_height=1, kernel_width=2, stride_size=2)
print(p.get_shape().as_list())
c = apply_conv(p, kernel_height=1, kernel_width=3, num_channels=8, depth=14)
print(c.get_shape().as_list())
p = apply_max_pool(c, kernel_height=1, kernel_width=2, stride_size=2)
print(p.get_shape().as_list())

shape = p.get_shape().as_list()
flat = tf.reshape(p, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(flat, f_weights), f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.add(tf.matmul(f, out_weights), out_biases)

cost_function = tf.reduce_mean(tf.square(y_ - Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    print("Training set MSE")
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, cost_function], feed_dict={X: batch_x, Y: batch_y})

        p_tr = session.run(y_, feed_dict={X: train_x})
        tr_mse = tf.reduce_mean(tf.square(p_tr - train_y))
        print(session.run(tr_mse))

    p_ts = session.run(y_, feed_dict={X: test_x})
    ts_mse = tf.reduce_mean(tf.square(p_ts - test_y))
    print("Test set MSE: %.4f" % session.run(ts_mse))