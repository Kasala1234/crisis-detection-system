import streamlit as st
import sys
import os
from PIL import Image

# Fix path for src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bert_predict import predict_bert
from src.image_model import predict_image

# Page config
st.set_page_config(page_title="Crisis Detection AI", layout="centered")

# 🎨 Custom UI Styling
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🚨 Smart Crisis Detection AI")
st.write("AI-powered detection using Deep Learning (BERT + CNN Fusion)")

# 📝 Text input
text = st.text_area("Enter a tweet")

# 🖼️ Image upload
uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

# 🚀 Analyze button
if st.button("Analyze"):

    text_pred, text_conf = 0, 0
    image_pred, image_conf = 0, 0

    # -------- TEXT ANALYSIS --------
    if text:
        text_pred, text_conf = predict_bert(text)
    else:
        st.warning("Please enter text")

    # -------- IMAGE ANALYSIS --------
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_pred, image_conf = predict_image(image)
    else:
        image_conf = 0

    # -------- FUSION (COMBINED DECISION) --------
    final_score = (text_conf + image_conf) / 2

    if text_pred == 1 or image_pred == 1:
        st.error(f"⚠️ Crisis Detected (Combined Confidence: {final_score:.2f})")
    else:
        st.success(f"✅ No Crisis (Confidence: {final_score:.2f})")

    # -------- DEBUG INFO (OPTIONAL BUT COOL) --------
    st.write("### 🔍 Model Details")
    st.write(f"Text Prediction: {text_pred}, Confidence: {text_conf:.2f}")
    st.write(f"Image Prediction: {image_pred}, Confidence: {image_conf:.2f}")