from tensorflow import keras
from keras.datasets import boston_housing
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print("train_data:", train_data.shape)
print("train_targets:", train_targets.shape)
print("test_data:", test_data.shape)

# data normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = keras.Sequential(
        [keras.layers.Dense(units=64, activation="relu"), keras.layers.Dense(units=64, activation="relu"),
         keras.layers.Dense(units=1)])

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    np.mean(all_scores)
