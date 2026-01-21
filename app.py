import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 UI & RADIANT FRONT-END
# -------------------------------
def apply_style_config():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=1920"
    
    st.markdown(f"""
    <style>
        /* Background and Global Font */
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
            color: white;
            font-family: 'Montserrat', sans-serif;
        }}

        /* Radiant Blue-to-Pink Title */
        .main-title {{
            font-weight: 900;
            font-size: 65px !important;
            text-align: center;
            background: linear-gradient(45deg, #00c6ff, #0072ff, #f781f3, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            filter: drop-shadow(0 0 10px rgba(238, 9, 121, 0.5));
        }}

        /* Floating Gradient Charms (Small & Radiant) */
        .charm {{
            position: fixed;
            font-size: 20px;
            z-index: 999;
            background: linear-gradient(45deg, #00c6ff, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: float-around 12s ease-in-out infinite alternate;
            pointer-events: none;
        }}

        @keyframes float-around {{
            0% {{ transform: translate(0, 0) rotate(0deg); opacity: 0.4; }}
            100% {{ transform: translate(150px, 100px) rotate(360deg); opacity: 0.8; }}
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: rgba(0, 0, 0, 0.8) !important;
            border-right: 1px solid #ee0979;
        }}

        /* Labels */
        .big-label {{
            font-size: 28px !important;
            font-weight: 700;
            color: #00c6ff;
            text-align: center;
            margin-bottom: 10px;
        }}

        /* Download Button Styling */
        .stDownloadButton>button {{
            background: linear-gradient(45deg, #00c6ff, #ee0979);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
        }}

        /* Progress Bar Color */
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, #00c6ff, #ee0979);
        }}
    </style>

    <!-- Floating Charms injected directly -->
    <div class="charm" style="top:15%; left:10%;">🎨</div>
    <div class="charm" style="top:45%; left:80%;">✨</div>
    <div class="charm" style="top:75%; left:20%;">🖌️</div>
    <div class="charm" style="top:10%; left:85%;">🌈</div>
    <div class="charm" style="top:60%; left:5%;">💎</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 FAST AI ENGINE (VGG19)
# -------------------------------
IMG_SIZE = 300 # Slightly bigger for clarity
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
CONTENT_LAYERS = ['block5_conv2']

@st.cache_resource
def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + CONTENT_LAYERS]
    return Model(vgg.input, outputs)

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess(img):
    img = img.numpy().reshape((IMG_SIZE, IMG_SIZE, 3))
    # Correcting mean subtraction for VGG19 clarity
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] # BGR to RGB
    return np.clip(img, 0, 255).astype('uint8')

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return result / num_locations

# -------------------------------
# 🚀 APP EXECUTION
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_style_config()

    # --- SIDEBAR: ALL INSPIRATIONS ---
    with st.sidebar:
        st.markdown("<h2 style='color:#ee0979;'>Gallery Reference</h2>", unsafe_allow_html=True)
        st.write("Use these as Style images for better results:")
        
        # Fixed Inspiration URLs
        styles_list = [
            ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
            ("Mosaic Art", "https://images.unsplash.com/photo-1597773330258-132d00133221?w=300"),
            ("Vibrant Abstract", "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=300")
        ]
        for name, url in styles_list:
            st.image(url, caption=name, use_column_width=True)

    # --- MAIN CANVAS ---
    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="big-label">🖼️ Subject</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("Upload content photo", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if c_file:
            c_img = Image.open(c_file).convert("RGB")
            st.image(c_img, use_column_width=True)

    with col2:
        st.markdown('<p class="big-label">🎨 Style</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("Upload style image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if s_file:
            s_img = Image.open(s_file).convert("RGB")
            st.image(s_img, use_column_width=True)

    # --- PROCESSING ---
    if c_file and s_file:
        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🎨 Mixing colors... (Ultra Fast Mode)", expanded=True):
                model = get_model()
                c_tensor = preprocess(c_img)
                s_tensor = preprocess(s_img)
                
                s_out = model(s_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                # Higher learning rate = Faster change
                opt = tf.optimizers.Adam(learning_rate=15.0)

                # FAST MODE: Only 40 iterations
                for i in range(41):
                    with tf.GradientTape() as tape:
                        outs = model(gen_img)
                        sl = tf.add_n([tf.reduce_mean((gram_matrix(outs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                        cl = tf.add_n([tf.reduce_mean((outs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                        # VERY high style weight for maximum change
                        loss = (5e0 * sl) + (1e4 * cl) 
                    
                    grad = tape.gradient(loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                
            final_art = deprocess(gen_img)
            
            # --- MINIMIZED OUTPUT ---
            st.markdown("<div style='text-align:center; padding-top:20px;'><h2 style='color:#f781f3;'>Masterpiece Complete</h2></div>", unsafe_allow_html=True)
            
            res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1])
            with res_col2:
                st.image(final_art, width=350)
                
                # --- DOWNLOAD FEATURE ---
                img_pil = Image.fromarray(final_art)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Masterpiece",
                    data=byte_im,
                    file_name="neural_art.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
