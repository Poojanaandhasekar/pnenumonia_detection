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
pneumonia-detector/
├── index.py                     # Main Streamlit app for prediction & chatbot
├── Dockerfile                   # Docker configuration to containerize the app
├── requirements.txt             # Python dependencies for app and models
├── models/
│   └── densenet_model_v3.keras  # Pre-trained CNN model (ensure this file is present)
├── README.md                    # Project documentation (you will add this)

▶️ Run the container
bash
Copy code
docker run -p 8501:8501 pneumonia-detector
Then open your browser and go to:
http://localhost:8501

🔍 Prerequisites
Python 3.8+

A trained model file at: models/densenet_model_v3.keras (must be provided)

Docker (optional, for containerized deployment)

🔄 Local Development
If not using Docker:

bash
Copy code
pip install -r requirements.txt
streamlit run index.py
📦 Deployment
You can deploy the containerized version using:

Render

Hugging Face Spaces (with Gradio support)

AWS/GCP/Azure (for cloud hosting)

✍️ License
This project is under the MIT License.
For educational and non-commercial use.

🙋‍♀️ Author
Pooja – DockerHub: poojaa796




