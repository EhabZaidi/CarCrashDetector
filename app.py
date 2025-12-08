"""
Streamlit Web App for Car Crash Detection
Upload an image and get a prediction: Yes (Crash) or No (No Crash)
"""
import streamlit as st
import torch
from PIL import Image
import os
from model_utils import load_model, predict_image

# Page configuration
st.set_page_config(
    page_title="Car Crash Detection",
    page_icon="🚗",
    layout="centered"
)

# Title and description
st.title("🚗 Car Crash Detection")
st.markdown("Upload an image to detect if it contains a car crash or collision.")
st.markdown("---")

# Load model (with caching to avoid reloading on every interaction)
@st.cache_resource
def get_model():
    """Load the model (cached to avoid reloading)"""
    model_path = 'car_crash_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first using cnnfin.ipynb to generate car_crash_model.pth")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = load_model(model_path, device=device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, device = get_model()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image file (JPG, PNG, or BMP)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Prediction button
        if st.button("🔍 Detect Crash", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Make prediction
                    result = predict_image(model, image, device=device, threshold=0.5)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    
                    # Show prediction with color coding
                    if result['prediction'] == 'Yes':
                        st.error(f"**Prediction: {result['prediction']} (Crash Detected)**")
                        st.warning("⚠️ This image appears to contain a car crash or collision.")
                    else:
                        st.success(f"**Prediction: {result['prediction']} (No Crash)**")
                        st.info("✅ This image does not appear to contain a car crash.")
                    
                    # Show confidence
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                    
                    # Show probability breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Crash Probability", f"{result['probability']*100:.2f}%")
                    with col2:
                        st.metric("No Crash Probability", f"{(1-result['probability'])*100:.2f}%")
                    
                    # Progress bar for visualization
                    st.progress(result['probability'])
                    st.caption(f"Crash probability: {result['probability']*100:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please make sure the image is valid and try again.")
    
    else:
        st.info("👆 Please upload an image file to get started.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) 
        trained on car crash images to detect collisions.
        
        **Model:** Custom CNN (from cnnfin.ipynb)
        **Accuracy:** ~82%
        """)
        
        st.markdown("---")
        st.header("📝 Instructions")
        st.markdown("""
        1. Upload an image file (JPG, PNG, or BMP)
        2. Click "Detect Crash" button
        3. View the prediction result
        """)
        
        st.markdown("---")
        st.header("⚙️ Technical Details")
        st.markdown(f"""
        **Device:** {device.upper()}
        **Model:** CNN with 4 convolutional layers
        **Input Size:** 224x224 pixels
        **Threshold:** 0.5
        """)

else:
    st.warning("⚠️ Model not loaded. Please ensure the model file exists.")
    st.markdown("""
    ### To use this app:
    1. Train the model using `cnnfin.ipynb`
    2. The notebook will save `car_crash_model.pth`
    3. Refresh this page to load the model
    """)

