import tensorflow as tf
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

# Load the MNIST dataset and normalize it
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Evaluate and save the model
loss, accuracy = model.evaluate(x_test, y_test)
model.save('handwritten_digit.keras')

# Function to classify the image
def classify_digit(model, image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Load grayscale
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = np.invert(img)  # Invert colors if necessary
    img = img / 255.0  # Normalize to [0, 1]
    img = img.reshape(1, 28, 28)  # Reshape to (1, 28, 28)

    # Use the model to predict
    prediction = model.predict(img)
    return prediction

# Function to resize the image for display
def resize_image(image, target_size):
    img = Image.open(image)
    resized_image = img.resize(target_size)
    return resized_image

# Streamlit setup
st.set_page_config('Digit Recognition', page_icon='🔢')
st.title('Handwritten Digit Recognition 🔢')
st.caption('by Hewan Alemayehu')

st.markdown(r'''This application recognizes digits from 0-9 from a PNG file with a resolution of 28x28 pixels.''')
st.subheader('Have fun giving it a try!!! 😊')

uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Run if an image is uploaded
if uploaded_image is not None:
    image_np = np.array(Image.open(uploaded_image))
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    cv2.imwrite(temp_image_path, image_np)

    resized_image = resize_image(uploaded_image, (300, 300))
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(resized_image)

    # Prediction button
    if st.button('Predict'):
        model = tf.keras.models.load_model('handwritten_digit.keras')
        prediction = classify_digit(model, temp_image_path)
        st.subheader('Prediction Result')
        st.success(f'The digit is probably a {np.argmax(prediction)}')

    os.remove(temp_image_path)  # Remove temporary file after prediction
