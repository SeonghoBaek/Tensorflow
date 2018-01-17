import tensorflow as tf

from utils import *
from autoencoder import *

batch_size = 100
batch_shape = (batch_size, 28, 28, 1)
num_visualize = 10

lr = 0.01
num_epochs = 50


def calculate_loss(original, reconstructed, loss_func='mse'):
    # Cross Entropy
    if loss_func == 'ce':
        eps = 1e-10
        cross_entropy = -1 * original * tf.log(reconstructed + eps) - 1 * (1 - original) * tf.log(1-reconstructed + eps)
        return tf.reduce_sum(cross_entropy) / batch_size

    # Mean Squared Error
    return tf.reduce_sum(tf.square(tf.subtract(reconstructed, original))) / batch_size


def train(dataset):
    input_image, reconstructed_image, phase_train = autoencoder(batch_shape)
    loss = calculate_loss(input_image, reconstructed_image, 'ce')
    #optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)


        try:
            saver.restore(session, 'model.ckpt')
        except:
            print('Restore failed')


        dataset_size = len(dataset.train.images)
        print("Dataset size:", dataset_size)
        num_iters = (num_epochs * dataset_size)/batch_size
        print("Num iters:", num_iters)

        for step in range(int(num_iters)):
            input_batch  = get_next_batch(dataset.train, batch_size)
            loss_val,  _, = session.run([loss, optimizer], feed_dict={input_image: input_batch, phase_train: True})
            if step % 1000 == 0:
                print("Loss at step", step, ":", loss_val)

        try:
            saver.save(session, 'model.ckpt')
        except:
            print('Save failed')

        test_batch = get_next_batch(dataset.test, batch_size)
        reconstruction = session.run(reconstructed_image,
                                     feed_dict={input_image: test_batch, phase_train: False})
        visualize(test_batch, reconstruction, num_visualize)


if __name__ == '__main__':
    dataset = load_dataset()
    train(dataset)