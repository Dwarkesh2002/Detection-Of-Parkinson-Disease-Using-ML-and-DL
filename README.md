# Parkinson's Disease Detection Using Deep Learning

## Overview

This project is a Parkinson’s Disease Detection system developed using Deep Learning and Machine Learning techniques.
The system analyzes spiral drawing images and predicts whether a person may have Parkinson’s Disease.

The project combines:

* CNN (VGG16) for feature extraction
* Triplet Loss for better feature learning
* KNN Classifier for final prediction

A Streamlit web application is also developed for easy user interaction.

---

# Features

* Upload spiral drawing images
* Automatic image processing
* Deep learning-based feature extraction
* Parkinson’s Disease prediction
* User-friendly Streamlit interface
* Clinical records storage

---

# Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Streamlit
* Scikit-learn

---

# Project Architecture

Input Image
↓
CNN (VGG16 Model)
↓
Feature Extraction
↓
Triplet Loss
↓
KNN Classifier
↓
Prediction Result

---

# Dataset

The dataset contains spiral drawing images collected for Parkinson’s Disease analysis.

The images are categorized into:

* Healthy
* Parkinson’s Disease

---

# Working Process

## Step 1: Image Upload

The user uploads a spiral drawing image through the Streamlit application.

## Step 2: Preprocessing

The image is resized and normalized for model input.

## Step 3: Feature Extraction

The CNN (VGG16) model extracts important image features.

## Step 4: Triplet Loss

Triplet Loss improves feature learning by reducing similarity errors.

## Step 5: Classification

The KNN classifier predicts whether the image belongs to:

* Healthy person
* Parkinson’s Disease patient

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

## Move to Project Folder

```bash
cd your-repository-name
```

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
streamlit run app.py
```

---

# Project Structure

```bash
├── app.py
├── model/
├── dataset/
├── clinical_records.csv
├── users.csv
├── requirements.txt
└── README.md
```

---

# Advantages

* Fast prediction
* Easy to use
* Non-invasive detection method
* Supports early disease identification

---

# Future Scope

* Improve model accuracy
* Add real-time camera detection
* Deploy on cloud platform
* Mobile application integration
* Multi-disease prediction support

---

# Screenshots

Add screenshots of:

* Home Page
* Upload Section
* Prediction Result
* Dashboard

---

# Author

Dwarkesh Girase

---

# Conclusion

This project demonstrates how Deep Learning and Machine Learning can help in the early detection of Parkinson’s Disease using image analysis techniques.
