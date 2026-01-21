import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 UI & BACKGROUND CONFIGURATION
# -------------------------------
def apply_artistic_theme():
    # High-quality Art Background Image
    # You can replace this URL with any high-res art image
    bg_img_url = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    st.markdown(f"""
    <style>
        /* Set Background Image for the whole app */
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
        }}

        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=Montserrat:wght@300;600&display=swap');

        /* Header Styling */
        .main-title {{
            font-family: 'UnifrakturMaguntia', cursive;
            font-size: 80px !important;
            text-align: center;
            color: #f1c40f;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
            margin-bottom: 0px;
        }}

        .sub-title {{
            font-family: 'Montserrat', sans-serif;
            text-align: center;
            letter-spacing: 5px;
            color: #ecf0f1;
            font-size: 14px;
            margin-bottom: 40px;
            text-transform: uppercase;
        }}

        /* Frosted Glass Cards */
        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
            margin-top: 20px;
        }}

        /* Button Styling */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(45deg, #f1c40f, #e67e22);
            color: black;
            font-weight: bold;
            border: none;
            padding: 15px;
            border-radius: 10px;
            transition: 0.4s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0 0 20px #f1c40f;
            color: white;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: rgba(0, 0, 0, 0.7) !important;
            backdrop-filter: blur(10px);
        }}

        /* Text Fixes */
        h1, h2, h3, p, span {{
            color: white !important;
        }}
        
        .stMarkdown div p {{
            font-family: 'Montserrat', sans-serif;
        }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI LOGIC (VGG19)
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
    img = img[:, :, ::-1] 
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
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
    s_outputs, c_outputs = model_outputs[:len(STYLE_LAYERS)], model_outputs[len(STYLE_LAYERS):]
    s_loss = tf.add_n([tf.reduce_mean((gram_matrix(s_outputs[i]) - style_features[i])**2) for i in range(len(STYLE_LAYERS))])
    c_loss = tf.add_n([tf.reduce_mean((c_outputs[i] - content_features[i])**2) for i in range(len(CONTENT_LAYERS))])
    return style_weight * s_loss + content_weight * c_loss

# -------------------------------
# 🚀 APP UI INTERFACE
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Gallery", layout="wide")
    apply_artistic_theme()

    # --- Header Section ---
    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Synthesizing Human Creativity with Machine Intelligence</p>', unsafe_allow_html=True)

    # --- Sidebar Parameters ---
    st.sidebar.markdown("## ⚙️ Studio Settings")
    iterations = st.sidebar.slider("Brush Strokes (Iterations)", 50, 500, 100)
    vibrancy = st.sidebar.select_slider("Style Vibrancy", options=["Soft", "Rich", "Explosive"], value="Rich")
    
    weight_map = {"Soft": (1e-3, 1e4), "Rich": (1e-2, 1e4), "Explosive": (1e-1, 1e4)}
    loss_weights = weight_map[vibrancy]

    # --- Workspace ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Subject Image")
        content_file = st.file_uploader("Upload the photo you want to transform", type=["jpg", "png", "jpeg"])
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            st.image(content_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 Artistic Style")
        style_file = st.file_uploader("Upload the style inspiration (e.g. Starry Night)", type=["jpg", "png", "jpeg"])
        if style_file:
            style_img = Image.open(style_file).convert("RGB")
            st.image(style_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Process Section ---
    if content_file and style_file:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🖌️ BEGIN ARTISTIC TRANSFORMATION"):
            with st.status("🎨 Mixing colors and applying AI brushstrokes...", expanded=True) as status:
                model = get_model()
                c_tensor = load_and_process_image(content_img)
                s_tensor = load_and_process_image(style_img)

                s_out = model(s_tensor)
                c_out = model(c_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = c_out[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                opt = tf.optimizers.Adam(learning_rate=5.0)

                for i in range(iterations):
                    with tf.GradientTape() as tape:
                        loss = compute_loss(model, loss_weights, gen_img, s_feats, c_feats)
                    grad = tape.gradient(loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                    if i % 20 == 0:
                        st.write(f"Refining details: {int((i/iterations)*100)}%")
                
                status.update(label="✅ Masterpiece Finished!", state="complete", expanded=False)

            final_image = deprocess_image(gen_img)
            
            # --- Display Final Result ---
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("<h2>Final Masterpiece</h2>", unsafe_allow_html=True)
            st.image(final_image, width=800)
            
            # Download Button
            res_pil = Image.fromarray(final_image)
            res_pil.save("artwork.png")
            with open("artwork.png", "rb") as f:
                st.download_button("📥 ADD TO COLLECTION (Download)", f, "my_art.png", "image/png")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
