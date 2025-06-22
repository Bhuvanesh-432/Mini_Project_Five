import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from PIL import Image

# ✅ Set page
st.set_page_config(page_title="Fish Image Classifier", layout="centered")

# ✅ Load model
@st.cache_resource
def load_fish_model():
    model = load_model("mobilenetv2_fish_model.h5")
    return model

model = load_fish_model()

# ✅ Class labels
class_labels = [
    'animal_fish',
    'animal_fish_bass',
    'black_sea_sprat',
    'gilt_head_bream',
    'hourse_mackerel',
    'red_mullet',
    'red_sea_bream',
    'sea_bass',
    'shrimp',
    'striped_red_mullet',
    'trout'
]

# ✅ Upload
st.title("🐟 Fish Image Classification")
st.markdown("Upload a fish image to predict its species.")

uploaded_file = st.file_uploader("Choose a .jpg or .png image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # ✅ Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ✅ Predict
    preds = model.predict(img_array)
    top_idx = np.argmax(preds)
    top_label = class_labels[top_idx]
    top_conf = preds[0][top_idx] * 100

    # ✅ Show Result
    st.success(f"🎯 Predicted Class: **{top_label}** ({top_conf:.2f}%)")

    # ✅ Show All Confidence Scores
    st.subheader("📊 Confidence Scores:")
    confidence_dict = {label: float(preds[0][i]) for i, label in enumerate(class_labels)}
    confidence_sorted = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

    st.bar_chart(list(confidence_sorted.values()), use_container_width=True)
    st.write({k: f"{v*100:.2f}%" for k, v in confidence_sorted.items()})
