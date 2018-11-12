# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper Libraries
import numpy as np
import os #delete image
import matplotlib.pyplot as plt
import cv2


print(tf.__version__)

cap = cv2.VideoCapture(0)
file_name = "temp.jpeg"
# train_images = np.array([])
train_images = []

# Capture Michael
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Michael', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(file_name, gray)
        temp = cv2.imread(file_name)
        train_images.append(temp)
        try:
            os.remove(file_name)
        except:
            pass
        print("captured")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# train_images = np.array(train_images)
# train_labels = np.zeros(len(train_images))
train_labels = [0]*len(train_images)

# Capture Not Michael
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Not Michael', gray)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
    if cv2.waitKey(1) & 0xFF == ord('v'):
        cv2.imwrite(file_name, gray)
        temp = cv2.imread(file_name)
        train_images.append(temp)
        try:
            os.remove(file_name)
        except:
            pass
        print("captured")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Append '1's the train_labels
while (len(train_labels)) != (len(train_images)):
    train_labels.append(1)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Preprocess
train_images = train_images[:, :, :, 0]
# train_labels = train_labels[:, :, :, 0]
train_images = train_images/255.0
train_labels = train_labels/255.0


# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(480, 640)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)



# New Webcam Image
cap = cv2.VideoCapture(0)
# pred_image = []
win_name = "Window"
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow(win_name, gray)
    cv2.imwrite(file_name, gray)
    pred_image = []
    temp = cv2.imread(file_name)
    pred_image.append(temp)
    pred_image = np.array(pred_image)
    pred_image = pred_image[:, :, :, 0]
    pred_image = pred_image / 255.0
    print("Model Predictions", model.predict(pred_image))
    if np.argmax(model.predict(pred_image)) == 0:
        # win_name = "Michael"
        print("Michael")
    else:
        # win_name = "Not Michael"
        print("Not Michael")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    try:
        os.remove(file_name)
    except:
        pass
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

