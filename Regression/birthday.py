import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
import os.path
import csv

def normalize_cols(m):
   col_max = m.max(axis=0)
   col_min = m.min(axis=0)
   return (m - col_min)/(col_max - col_min)


ops.reset_default_graph()
sess = tf.Session()

birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/' \
               '01_Introduction/07_Working_with_Data_Sources/' \
               'birthweight_data/birthweight.dat'

birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')

birth_header = birth_data[0].split('\t')
print(birth_header)

birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]

for i in range(len(birth_data[1:])):
   print(birth_data[i+1])

y_vals = np.array([x[0] for x in birth_data])
x_vals = np.array([x[2:] for x in birth_data])

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)

print('Train indices: \n', train_indices)

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

print('Test indices: \n', test_indices)

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

#########################################################

batch_size = 16
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[7, 1]))
#b = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.zeros(shape=[1]))
model_output = tf.add(tf.matmul(x_data, W), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
#my_opt = tf.train.GradientDescentOptimizer(0.01)
my_opt = tf.train.AdamOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc = []
test_acc = []

prediction = tf.round(tf.sigmoid(model_output))
correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(correct)

y_vals_train = np.expand_dims(y_vals_train, axis=1)

for i in range(160):
   #rand_index = np.random.choice(len(x_vals_train), size=batch_size)
   #rand_x = x_vals_train[rand_index]
   #rand_y = np.transpose([y_vals_train[rand_index]])
   #rand_y = np.expand_dims(y_vals_train[rand_index], axis=1)
   rand_index = np.random.choice(len(x_vals_train), size=len(x_vals_train))
   np.random.shuffle(rand_index)

   rand_x = x_vals_train[rand_index]
   rand_y = y_vals_train[rand_index]

   for j in range(int(len(x_vals_train) / batch_size)):
       if (j+1) * batch_size > len(rand_x):
           batch_idx = rand_index[j*batch_size:]
       else:
           batch_idx = rand_index[j*batch_size:(j + 1) * batch_size]
       xbatch_split = rand_x[batch_idx]
       ybatch_split = rand_y[batch_idx]

       sess.run([train_step], feed_dict={x_data: xbatch_split, y_target: ybatch_split})

   l, acc = sess.run([loss, accuracy], feed_dict={x_data: xbatch_split, y_target: ybatch_split})

   loss_vec.append(l)
   train_acc.append(acc)

   acc = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.expand_dims(y_vals_test, axis=1)})
   test_acc.append(acc)

   #if (i+1)%100 == 0:
   #print('Accuracy: ' + str(acc))
   #print(tw, tb)

plt.subplot(121)
plt.plot(loss_vec, 'b-')
plt.title('Cross Entropy Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
#plt.show()

plt.subplot(122)
plt.plot(train_acc, 'b-', label='Train Set Accuracy')
plt.plot(test_acc, 'r-', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()