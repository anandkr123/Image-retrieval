import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras


def top_3_index(a):
    """
    Find the index of top 3 values in an array
    :param a:
    :return:
    """
    top_3_idx = np.argsort(a)[-3:]
    return top_3_idx


# Load the test images
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Search a query image
query = x_test[random.randint(0, 9999)]
query_image_show = query

# Load model
loaded_model = tf.keras.models.load_model("image_retrieval")

# Create encodings of hidden representation
x_matrix_encodings = loaded_model.encoder(x_train[0:7000]).numpy()

matrixOfEncodings = x_matrix_encodings.reshape(7000, -1)

query = np.expand_dims(query, axis=0)
query = loaded_model.encoder(query).numpy()
query = query.reshape(1, -1)

listOfScores = []

# Create list of scores between query encoding and encoding database, XOR captures the non-overlapped pixels
for vector in matrixOfEncodings:
    x = (np.bitwise_xor(np.ma.make_mask(vector), np.ma.make_mask(query)))
    listOfScores.append(392 - np.count_nonzero(x))


# Top 3 matched images from the encodings database
top_3_max = top_3_index(listOfScores)

print("The two top max index", top_3_max[1], top_3_max[2], "and the MAX value is", listOfScores[top_3_max[1]],
      listOfScores[top_3_max[2]])


# Display the corresponding matched images
matched_image_1 = x_train[top_3_max[0]]
matched_image_2 = x_train[top_3_max[1]]
matched_image_3 = x_train[top_3_max[2]]
fig = plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(matched_image_1.reshape([28, 28]))

plt.subplot(2, 3, 2)
plt.imshow(matched_image_2.reshape([28, 28]))
plt.title('Top  3 Images matched from encodings database')

# The best matched image from the repository matrix
plt.subplot(2, 3, 3)
plt.imshow(matched_image_3.reshape([28, 28]))


# The corresponding test image that we are searching for
plt.subplot(2, 3, 5)
plt.imshow(np.squeeze(query_image_show))
plt.title('Query image')


plt.show()
