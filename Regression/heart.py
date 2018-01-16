import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize_cols(m):
   col_max = m.max(axis=0)
   col_min = m.min(axis=0)
   return (m - col_min)/(col_max - col_min)


def toFloat(val):
    if val == '"Absent"':
        return float(0)
    elif val == '"Present"':
        return float(1)
    else:
        return float(val)


sess = tf.Session()

f = open('heart.txt')
data = f.read()
f.close()

heart_data = data.split('\n')[1:]
heart_data = [[toFloat(x) for x in y.split('\t')] for y in heart_data]

#print(heart_data)

y_vals = np.array([x[-1] for x in heart_data])
x_vals = np.array([x[:-1] for x in heart_data])

#print(y_vals)
print(x_vals)

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)

#print('Train indices: \n', train_indices)

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

#print('Test indices: \n', test_indices)

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]

# Label Smoothing
def smooth(y, theta):
    if y == 0:
        y += theta
    else:
        y -= theta
    return y

#y_vals_train = [smooth(y, 0.2) for y in y_vals_train]

#print(y_vals_train)

x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

# Normalize
#x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
#x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

#########################################################

use_multi_layer = True

if use_multi_layer == True:
    batch_size = 32

    x_data = tf.placeholder(shape=[None, 9], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    W1 = tf.Variable(tf.truncated_normal(shape=[9, 9], stddev=0.1))
    b1 = tf.Variable(tf.zeros(shape=[9]))

    layer1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
    W2 = tf.Variable(tf.truncated_normal(shape=[9, 9], stddev=0.1))
    b2 = tf.Variable(tf.zeros(shape=[9]))

    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
    W3 = tf.Variable(tf.truncated_normal(shape=[9, 1], stddev=0.1))
    b3 = tf.Variable(tf.zeros(shape=[1]))

    model_output = tf.add(tf.matmul(layer2, W3), b3)
else:
    # Regression
    batch_size = 16
    x_data = tf.placeholder(shape=[None, 9], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    W = tf.Variable(tf.truncated_normal(shape=[9, 1], stddev=0.1))
    b = tf.Variable(tf.zeros(shape=[1]))
    model_output = tf.add(tf.matmul(x_data, W), b)

#total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
total_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_target, logits=model_output, label_smoothing=0.2)

#my_opt = tf.train.GradientDescentOptimizer(0.001)
my_opt = tf.train.AdamOptimizer(0.001)
train_step = my_opt.minimize(total_loss)
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc = []
test_acc = []

prediction = tf.round(tf.sigmoid(model_output))
correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(correct)

y_vals_train = np.expand_dims(y_vals_train, axis=1)

for i in range(2000):
   #rand_index = np.random.choice(len(x_vals_train), size=batch_size)
   #rand_x = x_vals_train[rand_index]
   #rand_y = np.transpose([y_vals_train[rand_index]])
   #rand_y = np.expand_dims(y_vals_train[rand_index], axis=1)
   rand_index = np.random.choice(len(x_vals_train), size=len(x_vals_train))
   np.random.shuffle(rand_index)

   rand_x = x_vals_train[rand_index]
   rand_y = y_vals_train[rand_index]

   for j in range(int(len(x_vals_train) / batch_size) + 1):
       if (j+1) * batch_size > len(rand_x):
           batch_idx = rand_index[j*batch_size:]
       else:
           batch_idx = rand_index[j*batch_size:(j + 1) * batch_size]
       xbatch_split = rand_x[batch_idx]
       ybatch_split = rand_y[batch_idx]

       sess.run([train_step], feed_dict={x_data: xbatch_split, y_target: ybatch_split})

   if i % 10 == 0:
    l, acc = sess.run([total_loss, accuracy], feed_dict={x_data: xbatch_split, y_target: ybatch_split})
    loss_vec.append(l)
    train_acc.append(acc)
    acc = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.expand_dims(y_vals_test, axis=1)})
    test_acc.append(acc)
    print('Epoch ' + str(i + 1) + ' Accuracy: ' + str(acc))

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