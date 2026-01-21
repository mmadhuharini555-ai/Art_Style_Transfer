import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 RADIANT SAKURA UI
# -------------------------------
def apply_ui_config():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=1920"
    
    st.markdown(f"""
    <style>
        /* Global Styles */
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }}

        /* Radiant Blue-to-Pink Title */
        .main-title {{
            font-weight: 900;
            font-size: 70px !important;
            text-align: center;
            background: linear-gradient(to right, #00c6ff, #f781f3, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
            filter: drop-shadow(0 0 15px rgba(247, 129, 243, 0.5));
        }}

        /* Flying Cherry Blossoms */
        .cherry-blossom {{
            position: fixed;
            top: -10%;
            z-index: 999;
            user-select: none;
            cursor: default;
            animation: fall linear infinite;
            color: #ffb7c5;
            font-size: 20px;
        }}

        @keyframes fall {{
            0% {{ transform: translateY(0vh) translateX(0px) rotate(0deg); opacity: 1; }}
            100% {{ transform: translateY(110vh) translateX(150px) rotate(360deg); opacity: 0; }}
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: rgba(15, 15, 15, 0.95) !important;
            border-right: 2px solid #ee0979;
        }}

        /* Button & Uploader Visibility Fix */
        /* Forces buttons and uploaders to be visible without needing hover */
        .stButton>button, .stFileUploader {{
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid #00c6ff !important;
            color: white !important;
            opacity: 1 !important;
            visibility: visible !important;
            display: block !important;
        }}

        .stButton>button {{
            background: linear-gradient(45deg, #00c6ff, #ee0979) !important;
            font-weight: bold !important;
            border: none !important;
            height: 50px;
            margin-top: 20px;
        }}

        .big-label {{
            font-size: 32px !important;
            font-weight: 800;
            color: #00c6ff;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(0, 198, 255, 0.5);
        }}

        /* Minimal Output Container */
        .output-container {{
            display: flex;
            justify-content: center;
            padding: 20px;
        }}
    </style>

    <!-- Cherry Blossom Petals -->
    <div class="cherry-blossom" style="left: 10%; animation-duration: 10s;">🌸</div>
    <div class="cherry-blossom" style="left: 25%; animation-duration: 15s; animation-delay: 2s;">🌸</div>
    <div class="cherry-blossom" style="left: 40%; animation-duration: 12s; animation-delay: 5s;">🌸</div>
    <div class="cherry-blossom" style="left: 60%; animation-duration: 18s; animation-delay: 1s;">🌸</div>
    <div class="cherry-blossom" style="left: 75%; animation-duration: 14s; animation-delay: 7s;">🌸</div>
    <div class="cherry-blossom" style="left: 90%; animation-duration: 11s; animation-delay: 3s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE (OPTIMIZED)
# -------------------------------
IMG_SIZE = 320 # Higher resolution for quality
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
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    return np.clip(img[:, :, ::-1], 0, 255).astype('uint8')

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return result / num_locations

# -------------------------------
# 🚀 APP EXECUTION
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_ui_config()

    # --- SIDEBAR: GALLERY INSPIRATIONS ---
    with st.sidebar:
        st.markdown("<h1 style='color:#ee0979; font-size:25px;'>Gallery Inspirations</h1>", unsafe_allow_html=True)
        
        # Fixed Image Links
        insp_styles = [
            ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
            ("Mosaic Art", "https://images.unsplash.com/photo-1543857778-c4a1a3e0b2eb?w=300"),
            ("Oil Abstraction", "https://images.unsplash.com/photo-1549490349-8643362247b5?w=300")
        ]
        for name, url in insp_styles:
            st.image(url, caption=name, use_column_width=True)

    # --- MAIN UI ---
    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("Give Content", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if c_file:
            st.image(Image.open(c_file), use_column_width=True)

    with col2:
        st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("Give Style", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if s_file:
            st.image(Image.open(s_file), use_column_width=True)

    # --- AI PAINTING ---
    if c_file and s_file:
        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🌸 Cherry blossoms falling... Style emerging...", expanded=True):
                model = get_model()
                c_img = Image.open(c_file).convert("RGB")
                s_img = Image.open(s_file).convert("RGB")
                
                c_tensor = preprocess(c_img)
                s_tensor = preprocess(s_img)
                
                # Pre-calculate features
                s_out = model(s_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                # Ultra High Learning Rate for instant change
                opt = tf.optimizers.Adam(learning_rate=20.0)

                # Shorter time: Only 30 iterations but aggressive weights
                for i in range(31):
                    with tf.GradientTape() as tape:
                        outs = model(gen_img)
                        # VERY high style weight (1e1) vs Content (1e4)
                        sl = tf.add_n([tf.reduce_mean((gram_matrix(outs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                        cl = tf.add_n([tf.reduce_mean((outs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                        loss = (1e1 * sl) + (1e4 * cl) 
                    
                    grad = tape.gradient(loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                
            final_art = deprocess(gen_img)
            
            # --- MINIMIZED DOWNLOADABLE RESULT ---
            st.markdown("<div style='text-align:center;'><h2 style='color:#f781f3;'>Final Masterpiece</h2></div>", unsafe_allow_html=True)
            
            # Columns to minimize output width
            r1, r2, r3 = st.columns([1, 1, 1])
            with r2:
                st.image(final_art, width=350)
                
                # Download
                img_pil = Image.fromarray(final_art)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button(label="📥 Download Art", data=buf.getvalue(), file_name="art.png", mime="image/png")

if __name__ == "__main__":
    main()
