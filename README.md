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

### Technologies Used
- **Frontend:** HTML, CSS
- **Backend:** Python (Flask / Streamlit)
- **Model Inference:** TensorFlow / Keras

---

⚙️ Installation & Setup
1️⃣ Clone the Repository
Bash

git clone [https://github.com/your-username/lung-cancer-detection.git](https://github.com/your-username/lung-cancer-detection.git)
cd lung-cancer-detection
2️⃣ Create a Virtual Environment (Recommended)
Bash

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
3️⃣ Install Dependencies
Bash

pip install -r requirements.txt
📂 Dataset Preparation
Download the dataset from Kaggle. Organize the dataset as follows:

Plaintext

datasets/
├── train/
│   ├── Benign/
│   ├── Malignant/
│   └── Normal/
└── val/
    ├── Benign/
    ├── Malignant/
    └── Normal/
🚀 Training the Model
Run the training script:

Bash

python train_model.py
This will:

Train the CNN model

Validate it on the validation dataset

Save the trained model to: models/cnn_model.keras

🔍 Single Image Prediction (CLI)
To test the model on a single CT image:

Bash

python predict.py
The script:

Loads the trained model

Preprocesses the image

Outputs prediction probabilities and the predicted class

🌐 Running the Web Application
Navigate to the web application directory:

Bash

cd web_app
Run the application:

Bash

python app.py
Open your browser and visit:

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
Upload a lung CT image to receive a prediction.

🧪 Important Notes
Image resolution 128×128 is used to preserve diagnostic CT features

Data augmentation and dropout help reduce overfitting

Slice-level predictions may differ from case-level diagnosis

Case-level majority voting can improve performance

⭐ Acknowledgements
Kaggle Community

IQ-OTH/NCCD dataset contributors

TensorFlow & Keras documentation
