import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout

IMAGE_SIZE = 250
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, "data\images")
CATEGORIES = os.listdir(datadir)
print(f"CATEGORIES: {CATEGORIES}\n")

# Define two empty list to contain image data
x, y = [], []

def preprocess():
    for category in CATEGORIES:
        path = os.path.join(datadir, category)
        classIndex = CATEGORIES.index(category)
        print(f"PATH: {path}")
        for imgs in tqdm(os.listdir(path)):
            img_arr = cv2.imread(os.path.join(path, imgs))
            resized_array = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
            x.append(resized_array)
            y.append(classIndex)
        print("\n")

preprocess()
cv2.destroyAllWindows()

# Split data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Convert and resize the data to a numpy array
X_train = np.array(X_train).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
Y_train = np.array(Y_train)
X_test = np.array(X_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
Y_test = np.array(Y_test)

batch_size = 32
epochs = 15

# Create the model architecture
model = Sequential()

model.add(Conv2D(64,(3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(16, activation="relu"))

model.add(Dense(len(CATEGORIES)))
model.add(Activation("softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

t1 = time.time()

# Fit the model
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1)
model.save("{}.h5".format(os.path.join(dirname) + "data/models"))

t2 = time.time()
print("Time taken: ",t2-t1)

print("Model evaluation: ")
validation_loss, validation_accuracy = model.evaluate(X_test, Y_test)