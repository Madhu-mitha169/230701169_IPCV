import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
closed_path = r"C:\Users\palani\Desktop\IPCV-2\train\Closed_Eyes"
open_path = r"C:\Users\palani\Desktop\IPCV-2\train\Open_Eyes"

IMG_SIZE = 64

data = []
labels = []

# Load Closed Eyes (label = 0)
for img in os.listdir(closed_path):
    img_path = os.path.join(closed_path, img)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image)
    labels.append(0)

# Load Open Eyes (label = 1)
for img in os.listdir(open_path):
    img_path = os.path.join(open_path, img)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image)
    labels.append(1)

# Convert to numpy
data = np.array(data)
labels = np.array(labels)

# Normalize
data = data / 255.0

# Reshape for CNN
data = data.reshape(-1, 64, 64, 1)

# One-hot encode labels
labels = to_categorical(labels, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("eye_model.h5")

print("? Model trained and saved as eye_model.h5")