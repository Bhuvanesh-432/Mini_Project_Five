# Mini_Project_Five

# 🐟 Multiclass Fish Image Classification

A deep learning project to classify fish species using Convolutional Neural Networks (CNN) and transfer learning. Includes a Streamlit web app for real-time predictions.

---

## 🚀 Project Overview

This project builds and evaluates multiple image classification models to recognize **11 fish species**. It uses:
- CNN from scratch
- Pretrained models: MobileNetV2, InceptionV3, NASNetMobile, EfficientNetB0
- Deployment using **Streamlit** web app

---

## 📁 Dataset

- 11 fish categories
- Images pre-organized into:
  - `data/train/`
  - `data/val/`
  - `data/test/`
- Each class contains ~250–500 images

---

## 🧠 Models Trained

| Model           | Accuracy | F1-score | Status       |
|-----------------|----------|----------|--------------|
| MobileNetV2     | ✅ 99%    | ✅ 99%    | 🏆 Best Model |
| InceptionV3     | ✅ 99%    | ✅ 99%    | Great        |
| NASNetMobile    | ✅ 97%    | ✅ 97%    | Fast & Clean |
| EfficientNetB0  | ❌ 16%    | ❌ 5%     | Failed to Learn |

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit**
- **Pillow**, **NumPy**, **Matplotlib**
- **Jupyter Notebook / VS Code**

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix

All models were evaluated on a holdout test set of 3,187 images.

---

## 🖥️ Streamlit Web App

### Features:
- Upload `.jpg/.png` fish image
- Predict species with top confidence
- View model confidence scores

### Run Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
