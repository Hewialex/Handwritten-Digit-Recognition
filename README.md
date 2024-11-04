![Project Screenshot](https://github.com/user-attachments/assets/c0fd9894-3a36-4ee5-9792-d3fe460643b4)

# Handwritten Digit Recognition

## Overview
This project is a Handwritten Digit Recognition application that uses a Convolutional Neural Network (CNN) to classify handwritten digits from 0 to 9. The model is built using TensorFlow and Keras, trained on the MNIST dataset, and deployed using Streamlit for a user-friendly interface.

## Features
- **Image Upload**: Users can upload an image of a handwritten digit in PNG, JPG, or JPEG format.
- **Digit Classification**: The application predicts the digit from the uploaded image and displays the result.
- **Interactive Interface**: Built with Streamlit, the app provides an intuitive user experience.

## Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Streamlit
- PIL (Python Imaging Library)

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Hewialex/Handwritten-Digit-Recognition]
   cd ["C:\Users\PC\Desktop\AIML"]

2. **Install the required packages**:
3. **Make sure you have Python installed, then install the necessary libraries using pip**:
   pip install tensorflow opencv-python numpy streamlit pillow
4. **Run the application**:
   Start the Streamlit application with the following command:
   streamlit run [Hand written digit recognition].py
   
**Usage**

- Once the application is running, navigate to the provided local URL (usually http://localhost:8501).
- Upload a handwritten digit image (28x28 pixels recommended).
- Click on the "Predict" button to classify the digit.
The predicted result will be displayed on the screen.

**Model Training**
The model was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits. The model architecture consists of:

- Input layer: Flattens the 28x28 pixel images.
- Two hidden layers: Each with 128 neurons and ReLU activation function.
- Output layer: 10 neurons with softmax activation function to predict probabilities for each digit.
- The model was trained for 3 epochs.

**Future Improvements**

- Enhance the model by experimenting with more complex architectures (e.g., Convolutional layers).
- Improve the user interface for better usability.
- Add features for continuous learning based on user feedback.

**Author**
- Hewan Alemayehu
