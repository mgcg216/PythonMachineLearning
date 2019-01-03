import os
# from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a model
# Let's build a simple model we'll use to demonstrate saving and loading weights.


# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
model.summary()

# Checkpoint callback usage
# Train the model and pass it the ModelCheckpoint callback:

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# model = create_model()
#
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback] # Pass callback to training
#           )

# This creates a single collection of TensorFlow checkoint files that are updated at the end of each epoch:

# Create a new, untrained model. When restoring amodel from only weights, you must have a model with the same
# architecture as the original model. Since it's the same model architecture, we can share weights despite that it's a
# different instance of the model.
# Now rebuild a fresh, untrained model and evaluate it on the test set. An untrained model will perform at chance levels
# (~10% accuracy):

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Then load the weights from the checkpoint, and re-evaluate:

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restore model, accuracy: {:5.2f}%".format(100*acc))

#  Checkpoint callback options
# The callback provides several options to give the resulting checkpoints unique names, and adjust the checkpoint
# frequency. Train a new model, and save uniquely named checkoint once every 5-epchs:

# include the epoch in the file name (use `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     # Save weights, every 5 Epochs.
#     period=5
# )
#
# model = create_model()
# model.fit(train_images, train_labels,
#           epochs=50, callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)

# Now, look at the resulting checkpoints and choose the latest one:

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Note: the default tensorflow format only saves the 5 most recent checkpoints.
# To test, rest the model and load the latest checkpoint:

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# What are these files?
# The above code stores the weights to a collection of checkpoint-formattted files that contain only the trained weights
# in a binary format. Checkpoints contain:
# + One or more chards that contain your model's weights.
# + An index file that indicates which weights are stored in a which shard.
# If you are only training amodel on  a single machine, you'll have one shard with the suffix: .datta-0000-of-00001

# Manually save weights
# Above you saw how to load the weights into a model
# Manually saving the weights is just as simple, use the Model.save_weights method.

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# # Restore the weights
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')
#
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Save the entire model
# The entire model can be saved to a file that contains the weights values, the model's configuration, and even the
# optimizer's configureation (depends on set up). This allows you to checkoint a model and resume training later-from
# the exact same state-without access to the original code.
# Saving a fully-functional model is very useful-you can load them in Tensorflow.js (HDf5, Saved Model) and then train
# and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite (HDF5, Saved Model)

# As an HDF5 file
# Keras provides a basic save format using the HDF5 standard. For our purposes, the saved model cna be treated as a
# single binary blob.

model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Now recreate the model from that file:

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# Check its accuracy:

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# This technique saves everything:
# + The weight values
# + The model's configuration (architecture)
# + The optimizer configuration
# Keras saves models by inspectin the architecture. Currently, it is not able to save TensorFlow optimizers
# (from tf.train). When using those you will need to re-compile the model after loading, and you will loose the state
# of the optimizer.

# As a saved_model
# Caution: This method of saving a tf.keras model is experimental and may change in futur vers8ions.
# Build a fresh model:

model = create_model()
model.fit(train_images, train_labels, epochs=5)

# create a saved_model:

saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
print(new_model)

# Run the restored model

# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%.format".format(100*acc))

