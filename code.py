import numpy as np
import cv2
from sklearn.model_selection import train_test_split
#import the modules

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# a function defined to load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
    return np.array(images)
# save the sketches and faces as sketches.npy and faces.npy and load them
sketches_folder = 'sketches.npy'
faces_folder = 'faces.npy'

sketches = load_images_from_folder(sketches_folder)
faces = load_images_from_folder(faces_folder)

# normalize them 
sketches = sketches / 255.0
faces = faces / 255.0

# if the sketches is in object type convert it to float64 
#to print the dtype of sktches and faces

print(sketches.dtype)
print(sketches.shape)
print(faces.dtype)
print(faces.shape)

import numpy as np

def normalize_images(images):
    """
    Normalize the pixel values of the input images to the range [0, 1].
    
    Args:
        images (numpy.ndarray): A numpy array containing the images to be normalized.
        
    Returns:
        numpy.ndarray: The normalized images.
    """
    normalized_sketches = (sketches - np.min(sketches)) / (np.max(sketches) - np.min(sketches))
    return normalized_sketches

import numpy as np
from sklearn.model_selection import train_test_split

# Assuming you have your dataset in numpy arrays called 'X' and the corresponding labels in 'y'
X_train, X_test, y_train, y_test = train_test_split(sketches, faces, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Dummy data loading for example purposes, replace with your actual data loading
# Replace the below lines with your actual data loading code
# sketches = [np.random.rand(64, 64) for _ in range(100)]  # Replace with actual data
# faces = [np.random.rand(64, 64) for _ in range(100)]  # Replace with actual data

# Ensure data is loaded correctly
if not isinstance(sketches, np.ndarray):
    sketches = np.array(sketches)
if not isinstance(faces, np.ndarray):
    faces = np.array(faces)

# Print initial shapes and types
print("Initial Sketches shape:", sketches.shape, "Type:", type(sketches))
print("Initial Faces shape:", faces.shape, "Type:", type(faces))

# Check if data is loaded correctly
if sketches.size == 0 or faces.size == 0:
    raise ValueError("Data arrays are empty. Please check your data loading process.")

# Ensure data is in the correct shape (samples, height, width, channels)
# For example, if images are 64x64 with 1 channel (grayscale)
if len(sketches.shape) == 3:  # Assuming sketches are (num_samples, height, width)
    sketches = sketches.reshape(-1, 128, 128, 1)  # Modify dimensions as needed
if len(faces.shape) == 3:  # Assuming faces are (num_samples, height, width)
    faces = faces.reshape(-1, 128, 128, 1)  # Modify dimensions as needed

# Print reshaped shapes
print("Reshaped Sketches shape:", sketches.shape)
print("Reshaped Faces shape:", faces.shape)

# Split data into training and validation sets
sketches_train, sketches_val, faces_train, faces_val = train_test_split(sketches, faces, test_size=0.2, random_state=42)

# Model definition and compilation (example model)
# Adjust the architecture as needed
combined_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128*128*1, activation='sigmoid'),  # Assuming output image size is 64x64 with 1 channel
    layers.Reshape((128, 128, 1))
])

combined_model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
combined_model.fit(sketches_train, faces_train,
                   epochs=50,
                   batch_size=32,
                   validation_data=(sketches_val, faces_val))
# 50 epochs are enough , if required more epochs can be added 
# this might cause overfitting , try to use early stop


# Evaluate the model
loss = combined_model.evaluate(sketches, [faces, faces])
print('Loss:', loss)

# Function to display images
def display_images(original, predicted, title):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(predicted, cmap='gray')
    axes[1].set_title(title)
    plt.show()

# Test with a new sketch
new_sketch = sketches[107]  # change the number based on the sketch labels 
predicted_face = combined_model.predict(new_sketch.reshape(1, 128, 128, 1))
display_images(new_sketch.reshape(128, 128), predicted_face[0].reshape(128, 128), 'Predicted Face')

# if colour images are required , use channel 3 instead of 1

