import random
import input_data
import ladder_network
import tensorflow as tf

print "Loading MNIST data"
mnist = input_data.read_data_sets(
    "MNIST_data/",
    one_hot=True,
    labeled_size=5000,
    validation_size=5000)

print
print mnist.train_unlabeled.num_examples, "unlabeled training examples"
print mnist.train_labeled.num_examples, "labeled training examples"
print mnist.validation.num_examples, "validation examples"
print mnist.test.num_examples, "test examples"


hyperparameters = {
  "learning_rate": 0.01,
  "noise_level": 0.2,
  "input_layer_size": 784,
  "class_count": 10,
  "encoder_layer_definitions": [
    (100, tf.nn.relu), # first hidden layer
    (50, tf.nn.relu),
    (10, tf.nn.softmax) # output layer
  ],
  "denoising_cost_multipliers": [
    1000, # input layer
    0.5, # first hidden layer
    0.1,
    0.1 # output layer
  ]
}

graph = ladder_network.Graph(**hyperparameters)

with ladder_network.Session(graph) as session:
  for step in xrange(1000):
    if step % 5 == 0:
      images, labels = mnist.train_labeled.next_batch(100)
      session.train_supervised_batch(images, labels, step)
    else:
      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(images, step)

    if step % 200 == 0:
      save_path = session.save()
      accuracy = session.test(
        mnist.validation.images, mnist.validation.labels, step)
      print
      print "Model saved in file: %s" % save_path
      print "Accuracy: %f" % accuracy
