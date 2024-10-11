# TITLE OF THE PROJECT : 
# Handwritten Digit Prediction using Convolutional Neural Networks 


# Objective : 
# -> To build a deep learning model that can automatically classify images of handwritten digits (0-9) using MNIST dataset. The aim is to achieve high accuracy in digit recognition


# Data Source :
# Dataset-> MNIST handwritten digit database
# Source -> The MNIST dataset is built into many machine learning libraries like keras and tensorflow . It contains 60,000 training images and 10,000 testing images, each a 28*28 grayscale image of digits 0-9.


# Import Library :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Import Data :
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Describe Data :
# ->The MNIST dataset contains 28*28 grayscale immags of digits 0 - 9
# -> Training Set : 60,0000 images
# -> Test Set : 10,000 images

# Data Visualization :
# Visualize some sample images from the dataset
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Digit: {y_train[i]}")
    plt.axis('off')
    plt.show()

# Data Processing :
# Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize pixel values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# Define Target Variables (y) and Features Variables (X) : 
# -> Target(y) : The digit label(0-9)
# -> Features(X) : The pixel values of the images
# One-hot encode the target labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


#Train Test Split : 
# -> The MNIST dataset comes pre spilt into training and test sets : 
# Training Set : x_train,y_train 
#Test Set : x_test,y_test


# Modeling : 
# Build the Convolutional Neural Network (CNN)
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)


# Model Evaluation : 
# Evaluate on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Prediction : 
# Predict on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Display the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
print('Confusion Matrix:')
print(cm)


# Explanation : 
# -> Model Architecture : The CNN uses convolutional layers to learn spatial features from images , followed by dense layers to classify the digits 
# -> Training and Evaluation : The model learns to recognize digit patterns through training , achieving good accuracy .
# Metrics : Confusion matrix and classification reports provide deeper insights into model performance for each digit


