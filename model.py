import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def show_image(image):
    """
    Display a sample image
    :param image:
    :return:
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(tf.squeeze(image))
    plt.gray()
    plt.show()


# Load the dataset
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Add noise to training image for the model to learn patterns by removing noise from thr image

noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)


class ImageRetrieval(tf.keras.Model):
    def __init__(self):
        super(ImageRetrieval, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, inputs, training=True, mask=None):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


model = ImageRetrieval()
model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError())

model.fit(x_train_noisy, x_train, epochs=10, shuffle=True, batch_size=128, validation_split=0.2)

model.encoder.summary()

model.decoder.summary()
model.save("image_retrieval")
