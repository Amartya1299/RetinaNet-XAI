import streamlit as st
from PIL import Image
from src.gradcam import generate_heatmap
import torch

st.title("RetinaNet-XAI: Diabetic Retinopathy Grading")
st.write("Upload a fundus image to get a diagnosis and visual explanation.")

uploaded_file = st.file_uploader("Choose a retinal image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        # In a real scenario, this would call your model.predict() and gradcam
        # For now, it triggers the visual reasoning pipeline [cite: 96]
        with st.spinner('Generating Heatmap...'):
            # Save temporary file for Grad-CAM
            temp_path = "temp_upload.png"
            image.save(temp_path)
            
            heatmap = generate_heatmap(temp_path, "outputs/checkpoints/best_model.pth")
            
            st.success("Analysis Complete!")
            st.subheader("Visual Explanation (Grad-CAM)")
            st.image(heatmap, caption='Heatmap: Red areas influenced the prediction.')