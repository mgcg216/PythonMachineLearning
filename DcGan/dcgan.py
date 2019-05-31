from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

print(tf.__version__)

import \
    glob  # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
import imageio  # makes gifs
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

# Load and prepare the dataset
"""
I will use the MNIST dataset to train the generator and the discriminator. The gnerator will generate handwritten digits
resembling the MNIST data.
"""

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFERED_SIZE = 60000  # representing the maximum number elments that will be buffered when prefecthing.
BATCH_SIZE = 256

# batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFERED_SIZE).batch(BATCH_SIZE)

# Create the models
# Both the generator and discriminator are defined using the Keras Sequential API.

# The Generator
"""
The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce to image from a seed (random noise). 
Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired images
size of 28x28x1. Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh
"""


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Use the (as yet untrained) generator to create an image

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

# The Discriminator
"""
The discriminator is a CNN-based image classifier
"""


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


"""
Use the 9as yet untrained) discriminator to classify the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images
"""

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# Define the loss and optimizer
# Define loss function and optimizers for both models.

# This method returns a helper fucntion to compute cross entoropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss
"""
This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) iamges to an array of 0s
"""


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Generator loss
"""
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classy the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator iptimizers are different sicne we will train two networks seperatorly

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save checkpoints
"""
This notebook also demonstrates how to save and restore models, whcih can be helpful in case of a long running training task is interrupted
"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Define the training loop

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise])

"""
The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes iamges ( prodcued by the generator). The loss is claculate for each of these models, and the gradients are used to upated the genrator and discriminator
"""


# Notice the use of 'tf.function'
# This annotation causes the function to be "compiledf".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generator_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generator_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_grdients(zip(gradient_of_discriminator, discriminator.trainable_variables))


# Generate and save images

def generate_and_save_images(model, epoch, test_input):
    # Notice 'training' is set to False
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        #         Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        #         Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    #     Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


# Train the model
"""
Call the train() method defined abocve to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images looks like random noise, As training progresses, the generated digits will look increasing real. After about 50 epochs, they resemeble MNIST digits. This may take about one minute/ epoch with the defaul settings on Colab
"""

# %%time
train(train_dataset, EPOCHS)

# Relu
"""
ReLU stands for rectified linear unit, and is type of activation function. Mathematically, it is defined as y = maxZ(0,x ).

ReLU is the most commonly usedactivation funciton in neural networks, especially in CNNs, If you are unsure what activation function to use in your network, ReLU is usually a good first choice.
"""

# How does ReLU compare
"""
ReLU is linear (identity) for all positive values, and zeros for all negative values. This means that:

- It's cheap to compute as there is no complicated math. The model can therefore take less time to train or run.

- It converges faster. Linerarity means that the slope doesn't plateau, or "saturate," when x gets large. It doesn't have the vanishing gradient problem suffered by other activation function like sigmoid or atanh.

- It's sparsley activated. Since ReLU is zero for all negative inputs, it's likeyly for any givne unit to not activate at all. This is often desirable (see below).
"""

# Sparsity
"""
Note: We are discussing model sparsity here. Data sparsisty (missing information) is different and usually bad.

Why is sparsity good? It makes intuitive sense if we think about the biological nerual network, which artificial ones try to imitate. While we have billions of nerons in our bodies , not all of thme fire all the time for evertythyihn we do. Instead they have different roles and are activated by different signals.

Sparsity resutls in concise models that often have better predictive power and less overfitting/noise. In a sparse network, it's more likely that nerons are actually processing meaninnful aspects of the problem. For example, in a model detecting cats in images, there may be a neron that can idetnify ears, which obviously shouldn't be activated if the image is about a building.

Finally, a sparse network is faster than a dense network, as there are fewer things to compute.
"""

# Dying ReLU
"""
The downside for being zero for all negative values is a problem called "dying ReLU."

A ReLU neron is "dead" if it's stuck in the negative side and always outputs 0. Because the slope of ReLU in the nagative range is also 0, once a neron gets negative, it's unlikeyly for it to recover. Such nerons are not playing anyrol in discriminating the input and is essentially uselss. Over the time you may end up with a large part of your network doing nothing.

You may be confused as of how this zero-slope section works in the first place. Remeber that a single step (in SGD, for example) involves multiple data points. As long as not all them are negative, we can still get a slope out of the ReLU. They dying problem is lekely to occure when learing rate is too high there is a large negative bias.

Lowering learning rates often mitigates the problem. If not, leaky ReLU and ELU are also good alternatives to try. They have slight slopes in the negative range, thereby preventing the issue.
"""

# Variants
"""
Leaky ReLU & Parametric ReLU(PReLU)
Leaky ReLU has a small slope for negative values, instead of altogether zero. For example, leaky ReLU may have y = 0.01x when x < 0

Parametric ReLU (PReLU) is a type of leaky ReLU that, instead of having a predetermined slope like 0.01, make it a parameter for the nerual network to figure out itself: y = ax when x <0

Leaky ReLU has two benefits:

- It fixes the "dying ReLU" problem, as it doesn't have zero-slope parts.

- It speeds up training. There is evidence that having the "mean activation" be close to 0 makes training faster. (It helps keep off-diagonal entries of the Fisther infromation matrix small, but you can safely ignore this.) Unlike ReLU, leaky ReLU is more "balanced," and may therefore learn faster.

Be aware that the result is not always consisten. Leaky Re_LU isn't always superior to plain ReLU, and should be sonidered only as an alternative.

Exponential Linear (ELU, SELU)

Similar to leaky ReLU, ELU has a small sloper for negative vlaues. Instead of a straight line, it uses a log curve.

It is designed to combine the good parts of ReLU and leakReLU--while it doesn't have the dying ReLU problem, it saturates for large negative values, allowing them to be essentially inactive.

It is sometimes c alled Scaled ELU (SELU) due to the constant factor a.

Concatenated ReLU (CReLU)

Concatenated ReLU has two outputs, one ReLU and one negative ReLU, concatenated together. In other words, for positive x it produces [x, 0], and for negative x it produces [0,x]. Because it has two outputs, CReLU doubles the output dimension. 

ReLU-6
You may run into ReLU-6 in some libraires, which is ReLU capped at 6. 

"""

# Activation functions
"""
Activation function are really important for a Artiicial Nerual Network to learn and make sense of something really complicated and Non-Lienar comples funcitonal mapping between the inputs and resposne variable. They introduce non-linear properties to out Network. Thier main purpose is to convert a input signal of a node in a A-NN to an output singal. That output singal now is used as a input in the next layer in the stack

Specifically in A-NN we do the sum of products of inputs(x) and their corresponding Weights(W) and apply a Activation function F(x) to it to get the output of that layer and feed it as an input to the next layer.
"""

# Convolution Transpose
"""
Convolution Tranpsoes applies convolution with a fractional stride. In other words spacing out the input values (wiht zeoes) to apply the filter over a region that's potentially smaller than the filter size.

As for the why one would want to use it. It cna be used as a sort of upsampling with learned weights as oppose to bilinear interpolation or some other fixed form of upsampling.
"""

# Upsampling
"""
Upsampling is the process of inserting zero-valued smaples between original smaples to increase the sampling rate.
"""

# Bilinear interpolation
"""
Bilinear interpolation is an extension of linear interpolation for interpolating funcitons of two variables (e.g., x and y) on a rectilinear 2D grid. The key idea is to perform interpolation first in one direction, and then again in the other direction
"""
