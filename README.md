# 🫁 Lung Cancer Detection using CNN (CT Scan Images)

This project implements a **Convolutional Neural Network (CNN)** to classify lung CT scan images into **Normal**, **Benign**, and **Malignant** categories.  
In addition to model training and evaluation, the project also includes a **web application** that allows users to upload CT images and receive real-time predictions.

---

## 📊 Dataset

**Dataset Name:** IQ-OTH/NCCD Lung Cancer Dataset  
**Source:** Kaggle  

🔗 **Dataset Link:** [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)

### Dataset Description
- Lung CT scan slices from real patient cases
- Three classes:
  - **Normal**
  - **Benign**
  - **Malignant**
- Labels are **case-level**, while images are **slice-level**
- Some benign cases contain visually normal slices (dataset limitation)

⚠️ *This may lead to certain benign slices being predicted as normal.*

---

## 🧠 Model Architecture

- Custom CNN built from scratch
- Input image size: **128 × 128**
- Architecture:
  - 3 × Convolutional layers (Conv2D + MaxPooling)
  - Fully connected dense layer
  - Dropout (0.5) to reduce overfitting
  - Softmax output layer (3 classes)
- Loss function: `categorical_crossentropy`
- Optimizer: `Adam`

---

## 🌐 Web Application

The project includes a **web-based application** for real-time lung cancer prediction.

### Web App Features
- Upload lung CT scan image
- Model inference using trained CNN
- Displays:
  - Predicted class
  - Confidence score
- Simple and user-friendly interface
