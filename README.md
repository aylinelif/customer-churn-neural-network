# Customer Churn Prediction using Neural Networks

This project implements a **binary classification model** to predict customer churn using a **feedforward neural network** built with TensorFlow and Keras.

## 📌 Project Overview

Customer churn prediction is a critical task for many businesses, as retaining existing customers is often more cost-effective than acquiring new ones.  
In this project, a neural network model is trained to predict whether a customer will churn based on numerical features.

## ⚙️ Data Preprocessing

- The dataset is split into training and test sets.
- Feature scaling is applied using **StandardScaler** to normalize the input features.
- The scaler is fitted on the training data and then applied to the test data to prevent data leakage.

## 🧠 Model Architecture

The model is built using the **Sequential** API and consists of:
- A fully connected hidden layer with **64 neurons** and ReLU activation
- Dropout layers with a rate of **0.3** to reduce overfitting
- A second hidden layer with **32 neurons** and ReLU activation
- A single-neuron output layer with **sigmoid activation** for binary classification

## 🏋️ Model Training

- Optimizer: **Adam**
- Loss function: **Binary Crossentropy**
- Metric: **Accuracy**
- Training is performed for up to **100 epochs**
- **Early Stopping** is used with a patience of **10 epochs**, monitoring validation loss
- The best model weights are restored automatically

## 📊 Results

The model achieves approximately **74–75% accuracy**, which is considered reasonable for churn prediction problems involving real-world, noisy tabular data.

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Jupyter Notebook

## 📎 Notes

This project is developed for educational purposes and demonstrates a clean and effective neural network pipeline, including preprocessing, regularization, and training control mechanisms.
