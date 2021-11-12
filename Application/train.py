# ProjectGurukul Face mask Detector

# import necessary packages
import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout

img_size = 100
dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, 'Data')
CATEGORIES = os.listdir(datadir)
print(CATEGORIES)

# Define two empty list to contain image data
x, y = [], []
   
def PreProcess():
    for category in CATEGORIES:
        path = os.path.join(datadir, category)
        classIndex = CATEGORIES.index(category)
        print(path)
        for imgs in tqdm(os.listdir(path)):
            img_arr = cv2.imread(os.path.join(path, imgs))
            
            # Resize the image
            resized_array = cv2.resize(img_arr, (img_size, img_size))
            cv2.imshow("images", resized_array)
            cv2.waitKey(1)
            resized_array = resized_array/255.0
            x.append(resized_array)
            y.append(classIndex)
            
PreProcess()
cv2.destroyAllWindows()

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Convert and resize the data to a numpy array
X_train = np.array(X_train).reshape(-1, img_size, img_size, 3)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(-1, img_size, img_size, 3)
y_test = np.array(y_test)

batch_size = 32
epochs = 15

# Create the model architecture
model = Sequential()

model.add(Conv2D(64,(3, 3), input_shape=(img_size, img_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(len(CATEGORIES)))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

t1 = time.time()

# Fit the model
model.fit(X_train, y_train, batch_size = batch_size, epochs=epochs, validation_split=0.3, verbose = 1)
model.save('{}.h5'.format(os.path.join(dirname) + "/model"))

t2 = time.time()
print('Time taken: ',t2-t1)

print("Model evaluation : ")
validation_loss, validation_accuracy = model.evaluate(X_test, y_test)