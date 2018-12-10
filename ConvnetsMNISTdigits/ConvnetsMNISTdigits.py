import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    layers.Flatten(),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

# print(model.summary())

mnist = tf.keras.datasets.mnist
to_cateogrical = tf.keras.utils.to_categorical


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_cateogrical(train_labels)
test_labels = to_cateogrical(test_labels)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
