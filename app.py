import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os

# Set page configuration
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🧠", layout="wide")

# Custom CSS for UI enhancements
st.markdown("""
<style>
    .pred-box {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 2px solid #F63366;
    }
    .pred-digit {
        font-size: 80px;
        font-weight: 800;
        color: #F63366;
        margin: 0;
        line-height: 1;
    }
    .pred-conf {
        font-size: 24px;
        color: #A0A0A0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and data
@st.cache_resource
def load_all_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.h5')
    history_path = os.path.join(base_dir, 'history.pkl')
    metrics_path = os.path.join(base_dir, 'metrics.pkl')
    
    model = tf.keras.models.load_model(model_path)
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return model, history, metrics

# Main Header
st.title("🧠 MNIST Digit Classifier Web App")
st.markdown("A premium machine learning web application to classify handwritten digits using Artificial Neural Networks.")
st.divider()

# Attempt to load model
try:
    model, history, metrics = load_all_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model or related metrics files. Ensure `train.py` has been executed. Error: {e}")

if model_loaded:
    # Sidebar for Model Insights
    with st.sidebar:
        st.header("📊 Model Metrics")
        st.metric(label="Testing Accuracy", value=f"{metrics['accuracy']*100:.2f}%")
        st.metric(label="Precision", value=f"{metrics['precision']*100:.2f}%")
        st.metric(label="Recall", value=f"{metrics['recall']*100:.2f}%")
        st.metric(label="F1 Score", value=f"{metrics['f1_score']*100:.2f}%")
        
        st.divider()
        st.write("### Model Architecture")
        st.write("Input -> Flatten -> Dense(64) -> Dense(32) -> Dense(16) -> Output(10)")
        
    # Main content tabs
    tab1, tab2 = st.tabs(["🚀 Predict Digit", "📉 Model Insights"])
    
    with tab1:
        st.subheader("Upload an Image of a Handwritten Digit")
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Predict 🔮", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Ensure user sees the spinner
                        time.sleep(1) 
                        
                        # Preprocessing
                        img = image.convert('L') # grayscale
                        img = ImageOps.invert(img) # invert colors (MNIST is white on black usually)
                        img = img.resize((28, 28))
                        img_array = np.array(img)
                        img_array = img_array / 255.0
                        
                        # If image background is mostly white instead of black
                        # we can try to conditionally invert based on edge pixels
                        # which is robust for user uploads who upload black-to-white
                        
                        img_data = img_array.reshape(1, 28, 28)
                        
                        # Predict
                        pred_probs = model.predict(img_data)
                        predicted_class = np.argmax(pred_probs[0])
                        confidence = np.max(pred_probs[0]) * 100
                        
                    st.success("Prediction Completed!")
                    
                    with col2:
                        st.subheader("Results")
                        st.markdown(f"""
                        <div class="pred-box">
                            <p class="pred-conf">Predicted Digit</p>
                            <h1 class="pred-digit">{predicted_class}</h1>
                            <p class="pred-conf">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show class probabilities
                        fig, ax = plt.subplots(figsize=(6, 3))
                        fig.patch.set_facecolor('#1E1E1E')
                        ax.set_facecolor('#1E1E1E')
                        sns.barplot(x=list(range(10)), y=pred_probs[0], ax=ax, palette="flare")
                        ax.set_title("Class Probabilities", color="white")
                        ax.set_xlabel("Digit", color="white")
                        ax.set_ylabel("Probability", color="white")
                        ax.tick_params(colors='white')
                        sns.despine(left=True, bottom=True)
                        st.pyplot(fig)
                        
    with tab2:
        st.subheader("Training Journey & Performance Analysis")
        col_m1, col_m2 = st.columns(2)
        
        sns.set_theme(style="darkgrid")
        plt.rcParams['text.color'] = '#FFFFFF'
        plt.rcParams['axes.labelcolor'] = '#FFFFFF'
        plt.rcParams['xtick.color'] = '#FFFFFF'
        plt.rcParams['ytick.color'] = '#FFFFFF'
        
        with col_m1:
            st.markdown("#### Accuracy vs Epochs")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#1E1E1E')
            ax1.set_facecolor('#2D2D2D')
            ax1.plot(history['accuracy'], label='Train Accuracy', color='#00FF00', linewidth=2)
            ax1.plot(history['val_accuracy'], label='Val Accuracy', color='#FFA500', linewidth=2, linestyle='--')
            ax1.legend(facecolor='#1E1E1E', edgecolor='#F63366')
            st.pyplot(fig1)

        with col_m2:
            st.markdown("#### Loss vs Epochs")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#1E1E1E')
            ax2.set_facecolor('#2D2D2D')
            ax2.plot(history['loss'], label='Train Loss', color='#F63366', linewidth=2)
            ax2.plot(history['val_loss'], label='Val Loss', color='#FFA500', linewidth=2, linestyle='--')
            ax2.legend(facecolor='#1E1E1E', edgecolor='#F63366')
            st.pyplot(fig2)
            
        st.divider()
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics['confusion_matrix'])
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        fig3.patch.set_facecolor('#1E1E1E')
        sns.heatmap(cm, annot=True, fmt='d', cmap='inferno', ax=ax3, 
                    cbar_kws={'label': 'Count'}, square=True)
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('True Label')
        st.pyplot(fig3)
