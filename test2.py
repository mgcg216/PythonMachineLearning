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
capture = False
file_name = "temp.jpeg"
# train_images = np.array([])
train_images = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Hello World', gray)
    # if capture is True:
    #     print("capturing")
    #     train_images.append(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # if capture is False:
        #     capture = True
        # if capture is True:
        #     capture = False
        cv2.imwrite(file_name, gray)
        temp = cv2.imread(file_name)
        # temp = tf.image.decode_image(file_name)
        # temp = keras.preprocessing.image.load_img(file_name)
        # np.append(train_images, temp)
        train_images.append(temp)
        # print("temp", temp)
        # print("train images", train_images)
        try:
            os.remove(file_name)
        except:
            pass

        print("captured")
#print("train image shape")
#print(train_images.shape)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = np.array(train_images)
train_labels = np.zeros(len(train_images))



#train_labels = [0]*len(train_images)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Preprocess
train_images = train_images[:, :, :, 0]
# train_labels = train_labels[:, :, :, 0]
train_images = train_images/255.0
train_labels = train_labels/255.0


# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(480, 640)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)



# New Webcam Image
cap = cv2.VideoCapture(0)

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
    pred_image = np.array(file_name)
    if np.argmax(model.predict(pred_image)) == 0:
        win_name = "Michael"
    else:
        win_name = "Not Michael"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

