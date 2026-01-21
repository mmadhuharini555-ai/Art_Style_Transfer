import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 1. Custom Eye-Catching Styling
# -------------------------------
def local_css():
    st.markdown("""
    <style>
        /* Main background and fonts */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Roboto:wght@300;400&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
        }
        
        .main {
            background-color: #f5f5f5;
        }

        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: #2c3e50;
        }

        /* Styling the upload boxes */
        .stFileUploader {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 10px;
        }

        /* Customizing buttons */
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
            transition: 0.3s;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #e67e22;
            color: white;
            transform: scale(1.02);
        }

        /* Card-like containers for images */
        .img-card {
            background: white;
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Logic Constants & Models
# -------------------------------
IMG_SIZE = 224
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def load_and_process_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(img):
    img = img.numpy().reshape((IMG_SIZE, IMG_SIZE, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] # BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

@st.cache_resource
def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + CONTENT_LAYERS]
    return Model(vgg.input, outputs)

def compute_loss(model, loss_weights, generated_image, style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(generated_image)
    generated_style = model_outputs[:len(STYLE_LAYERS)]
    generated_content = model_outputs[len(STYLE_LAYERS):]

    style_loss = tf.add_n([tf.reduce_mean((gram_matrix(generated_style[i]) - style_features[i])**2) for i in range(len(STYLE_LAYERS))])
    content_loss = tf.add_n([tf.reduce_mean((generated_content[i] - content_features[i])**2) for i in range(len(CONTENT_LAYERS))])
    
    return style_weight * style_loss + content_weight * content_loss

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Studio", layout="wide")
    local_css()
    
    # --- Header Section ---
    st.markdown("<h1 style='text-align: center;'>🎨 Neural Art Studio</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>Transform your photos into masterpieces using Convolutional Neural Networks</p>", unsafe_allow_html=True)
    st.divider()

    # --- Sidebar Controls ---
    st.sidebar.header("🖌️ Studio Settings")
    iterations = st.sidebar.slider("Number of Iterations", 50, 500, 100)
    style_intensity = st.sidebar.select_slider("Style Intensity", options=["Low", "Medium", "High", "Extreme"], value="Medium")
    
    # Map intensity to weights
    intensity_map = {"Low": 1e-4, "Medium": 1e-2, "High": 1e-1, "Extreme": 1.0}
    style_weight = intensity_map[style_intensity]
    content_weight = 1e4

    # --- Upload Section ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Content Image")
        content_file = st.file_uploader("Choose a base photo...", type=["jpg", "png"], key="content")
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            st.image(content_img, use_column_width=True)

    with col2:
        st.markdown("### 🎨 Style Image")
        style_file = st.file_uploader("Choose an art style...", type=["jpg", "png"], key="style")
        if style_file:
            style_img = Image.open(style_file).convert("RGB")
            st.image(style_img, use_column_width=True)

    # --- Processing Section ---
    if content_file and style_file:
        st.divider()
        if st.button("✨ Generate Masterpiece"):
            model = get_model()
            
            content_tensor = load_and_process_image(content_img)
            style_tensor = load_and_process_image(style_img)
            
            # Get target features
            style_outputs = model(style_tensor)
            content_outputs = model(content_tensor)
            style_features = [gram_matrix(out) for out in style_outputs[:len(STYLE_LAYERS)]]
            content_features = content_outputs[len(STYLE_LAYERS):]

            generated_image = tf.Variable(content_tensor, dtype=tf.float32)
            optimizer = tf.optimizers.Adam(learning_rate=5.0)

            # Progress Tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            image_placeholder = st.empty()

            for i in range(iterations):
                with tf.GradientTape() as tape:
                    loss = compute_loss(model, (style_weight, content_weight), generated_image, style_features, content_features)
                
                grad = tape.gradient(loss, generated_image)
                optimizer.apply_gradients([(grad, generated_image)])
                
                # Update UI periodically
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / iterations)
                    status_text.text(f"Processing... Iteration {i}/{iterations}")
                    # Optional: Show preview every 50 iterations
                    # preview = deprocess_image(generated_image)
                    # image_placeholder.image(preview, caption="Refining details...", width=400)

            final_image = deprocess_image(generated_image)
            
            st.success("✅ Transformation Complete!")
            
            # --- Final Result Display ---
            res_col1, res_col2, res_col3 = st.columns([1, 6, 1])
            with res_col2:
                st.markdown("<h2 style='text-align: center;'>Final Masterpiece</h2>", unsafe_allow_html=True)
                st.image(final_image, use_column_width=True)
                
                # Download button
                result_pil = Image.fromarray(final_image)
                result_pil.save("output.png")
                with open("output.png", "rb") as file:
                    st.download_button(
                        label="📥 Download Artwork",
                        data=file,
                        file_name="masterpiece.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()