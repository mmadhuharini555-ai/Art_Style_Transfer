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
    bg_img_url = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    st.markdown(f"""
    <style>
        /* Set Background Image */
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        /* Header Styling */
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 60px !important;
            text-align: center;
            color: #ffffff;
            margin-bottom: 0px;
            text-shadow: 0px 0px 20px rgba(0,255,255,0.5);
        }}

        .sub-title {{
            font-family: 'Montserrat', sans-serif;
            text-align: center;
            letter-spacing: 4px;
            color: #bdc3c7;
            font-size: 14px;
            margin-bottom: 50px;
            text-transform: uppercase;
        }}

        /* FIXING THE UPLOAD BOXES */
        /* This targets the actual Streamlit File Uploader container */
        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            padding: 20px;
            color: white !important;
        }}

        /* Ensuring text inside uploader is bright and visible */
        [data-testid="stFileUploader"] section {{
            color: white !important;
        }}
        
        [data-testid="stFileUploader"] label {{
            color: #f1c40f !important; /* Golden labels */
            font-size: 1.5rem !important;
            font-weight: bold !important;
            margin-bottom: 15px !important;
        }}

        /* Button Styling */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            font-weight: bold;
            border: none;
            padding: 20px;
            border-radius: 50px;
            transition: 0.4s;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 20px;
        }}

        .stButton>button:hover {{
            box-shadow: 0 0 30px rgba(0, 198, 255, 0.6);
            transform: translateY(-3px);
        }}

        /* Remove the streamlit branding */
        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI LOGIC
# -------------------------------
IMG_SIZE = 224
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
CONTENT_LAYERS = ['block5_conv2']

@st.cache_resource
def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + CONTENT_LAYERS]
    return Model(vgg.input, outputs)

def load_and_process_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_image(img):
    img = img.numpy().reshape((IMG_SIZE, IMG_SIZE, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] 
    return np.clip(img, 0, 255).astype('uint8')

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return result / num_locations

# -------------------------------
# 🚀 APP UI
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Gallery", layout="wide")
    apply_artistic_theme()

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Synthesizing Human Creativity with Machine Intelligence</p>', unsafe_allow_html=True)

    # Workspace
    col1, col2 = st.columns(2)

    with col1:
        # We use the label of the uploader to show the text you wanted
        content_file = st.file_uploader("🖼️ Give your content image", type=["jpg", "png", "jpeg"])
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            st.image(content_img, caption="Subject Photo", use_column_width=True)

    with col2:
        style_file = st.file_uploader("🎨 Give your style image", type=["jpg", "png", "jpeg"])
        if style_file:
            style_img = Image.open(style_file).convert("RGB")
            st.image(style_img, caption="Style Inspiration", use_column_width=True)

    if content_file and style_file:
        if st.button("🖌️ Create Masterpiece"):
            with st.status("🎨 Applying AI Brushstrokes...", expanded=True):
                model = get_model()
                c_tensor = load_and_process_image(content_img)
                s_tensor = load_and_process_image(style_img)

                s_out = model(s_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                opt = tf.optimizers.Adam(learning_rate=5.0)

                # Short loop for demonstration (you can increase iterations)
                for i in range(101):
                    with tf.GradientTape() as tape:
                        outputs = model(gen_img)
                        s_loss = tf.add_n([tf.reduce_mean((gram_matrix(outputs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                        c_loss = tf.add_n([tf.reduce_mean((outputs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                        total_loss = (1e-2 * s_loss) + (1e4 * c_loss)
                    
                    grad = tape.gradient(total_loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                
            final_image = deprocess_image(gen_img)
            st.markdown("<div style='text-align:center;'><h2>Final Artwork</h2></div>", unsafe_allow_html=True)
            st.image(final_image, use_column_width=True)

if __name__ == "__main__":
    main()
