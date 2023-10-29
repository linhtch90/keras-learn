from tensorflow import keras
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print("train_data len:", len(train_data))
print("train_labels len:", len(train_labels))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# def one_hot_encoding(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results
#
#
# y_train = one_hot_encoding(train_labels)
# y_test = one_hot_encoding(test_labels)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = keras.Sequential(
    [keras.layers.Dense(units=64, activation="relu"), keras.layers.Dense(units=64, activation="relu"),
     keras.layers.Dense(units=46, activation="softmax")])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=9, batch_size=512)

results = model.evaluate(x_test, y_test)
print("results:", results)
