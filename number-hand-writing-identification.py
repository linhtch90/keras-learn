from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(">>> train_images", train_images.shape)
print(">>> test_images", test_images.shape)

model = keras.Sequential(
    [layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")]
)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=100)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(">>> predictions at index 0:", predictions[0].argmax())
print(">>> predictions accuracy at index 0:", predictions[0].max())
print(">>> actual value at index 0:", test_labels[0])

evaluate_result = model.evaluate(test_images, test_labels)
test_loss, test_accuracy = evaluate_result
print(">>> test_loss:", test_loss)
print(">>> test_accuracy:", test_accuracy)
