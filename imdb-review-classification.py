import numpy as np
from keras import layers
from keras.datasets import imdb
from tensorflow import keras

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(nb_words=10000)

print("train_data length", len(train_data))
print("train_label length", len(train_labels))
print("train_data item length", len(train_data[0]))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(train_labels).astype("float32")

model = keras.Sequential(
    [layers.Dense(16, activation="relu"), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])


model.compile