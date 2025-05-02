#
# 🫁 Pneumonia Detection and Medical Q&A App

This is a Streamlit-based web application that uses a Convolutional Neural Network (CNN) to detect **Pneumonia** from Chest X-ray images. It also integrates a Hugging Face **LLM** (Language Model) to explain the diagnosis in simple terms and answer medical questions via a chatbot.

---

## 🚀 Features

- Upload and analyze Chest X-ray images
- Predict **Pneumonia** using a trained CNN model
- Generate simple AI-powered explanations for patients
- Ask health-related questions using a medical Q&A chatbot
- Download a medical report

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend Models**: 
  - CNN (Keras/TensorFlow)
  - Hugging Face Transformers (DistilGPT2, Roberta-based Q&A)
- **Containerization**: Docker

---

## 📁 Project Structure
.
├── index.py # Streamlit app
├── Dockerfile # Docker container setup
├── requirements.txt # Python dependencies
└── models/
└── densenet_model_v3.keras # Pre-trained CNN model (user-provided)


