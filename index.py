import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set model path
MODEL_PATH = "models/densenet_model_v3.keras"

# Load the CNN model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

# Load Hugging Face LLM pipeline (PyTorch backend)
try:
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id)
    llm = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
except Exception as e:
    st.warning(f"⚠️ Failed to load LLM: {e}")
    llm = None

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("🫁 Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to predict if **Pneumonia** is present and get an AI-generated explanation.")

# Upload image
uploaded_file = st.file_uploader("📂 Choose a Chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    st.subheader("🧪 Raw Prediction:")
    st.write(prediction)

    # Interpret prediction
    if prediction.shape[-1] == 1:
        is_normal = prediction[0][0] > 0.5
        result = "🟢 Normal" if is_normal else "🔴 Pneumonia Detected"
    else:
        class_labels = ["🟢 Normal", "🔴 Pneumonia Detected"]
        result = class_labels[np.argmax(prediction)]
        is_normal = result == "🟢 Normal"

    st.subheader("✅ Final Result:")
    st.markdown(f"### {result}")

    # Generate explanation with Hugging Face LLM
    if llm:
        if is_normal:
            input_text = "The model detected the chest X-ray as Normal. Explain this to a patient in simple words."
        else:
            input_text = "The model detected signs of Pneumonia in the chest X-ray. Explain this to a patient in simple words."

        try:
            generated = llm(input_text, max_length=100, num_return_sequences=1)
            explanation = generated[0].get("generated_text", "No explanation generated.")
        except Exception as e:
            explanation = f"❌ LLM generation failed: {e}"
    else:
        explanation = "⚠️ LLM not available."

    st.subheader("🧠 Doctor's Explanation (AI-powered):")
    st.write(explanation)

    # 📄 Add downloadable medical report
    medical_report = f"""
    🩺 Pneumonia Detection Report

    Patient Image: {uploaded_file.name}

    Prediction Result: {result}
    Raw Prediction: {prediction.tolist()}

    Doctor's Explanation:
    {explanation}

    Advice:
    If you are feeling unwell, consult a healthcare provider immediately.
    """
    st.download_button(
        label="📥 Download Medical Report",
        data=medical_report,
        file_name="pneumonia_report.txt",
        mime="text/plain"
    )

else:
    st.info("📂 Please upload a chest X-ray image to get started.")

# ----------------------------
# 💬 Medical Q&A Chatbot
# ----------------------------
st.markdown("---")
st.subheader("💬 Ask a Medical Question")

qa_input = st.text_input("🗨️ Ask your medical question (e.g. How to cure pneumonia?)")

if qa_input:
    try:
        from transformers import pipeline as qa_pipeline
        qa = qa_pipeline("question-answering", model="deepset/roberta-base-squad2")

        context = """
        Pneumonia is a lung infection that causes inflammation in the air sacs, often due to bacteria, viruses, or fungi.
        It can cause cough with phlegm, fever, chills, and difficulty breathing. Mild cases can be treated at home with rest,
        fluids, and antibiotics if bacterial. Severe cases may need hospitalization, oxygen therapy, or IV antibiotics.
        Pneumonia is more dangerous for infants, elderly, and people with weak immune systems.
        """

        response = qa(question=qa_input, context=context)
        st.markdown(f"**💡 Answer:** {response['answer']}")
    except Exception as e:
        st.error(f"❌ Chatbot failed: {e}")
