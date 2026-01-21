import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 UI & DECORATION CONFIGURATION
# -------------------------------
def apply_art_studio_theme():
    bg_img_url = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Radiant Title */
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 900;
            font-size: 80px !important;
            text-align: center;
            background: linear-gradient(to right, #00f2fe, #4facfe, #f1c40f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0px 0px 15px rgba(79, 172, 254, 0.5));
            margin-bottom: 40px;
        }}

        /* Massive Labels */
        .big-label {{
            font-size: 38px !important;
            font-weight: 800 !important;
            color: #4facfe !important;
            text-align: center;
            margin-bottom: 15px;
            text-transform: uppercase;
        }}

        /* Artistic Charms (Left Side) */
        .charm {{
            font-size: 40px;
            margin-bottom: 20px;
            text-align: center;
            filter: drop-shadow(0 0 10px #f1c40f);
            animation: float 3s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-15px); }}
            100% {{ transform: translateY(0px); }}
        }}

        /* Inspiration Gallery (Right Side) */
        .gallery-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 5px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }}
        
        .gallery-text {{
            font-size: 12px;
            color: #bdc3c7;
            margin-top: 5px;
        }}

        /* Uploaders */
        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.03);
            border: 2px dashed rgba(79, 172, 254, 0.3);
            border-radius: 20px;
        }}

        /* Hide Sidebar and Menus */
        [data-testid="stSidebar"] {{display: none;}}
        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE
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

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
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
# 🚀 APP UI
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Gallery", layout="wide")
    apply_art_studio_theme()

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    # LAYOUT: Charms | Main Workspace | Inspiration Gallery
    col_charms, col_main, col_gallery = st.columns([0.8, 7, 2])

    # 1. Left Charms
    with col_charms:
        st.markdown('<div class="charm">🎨</div>', unsafe_allow_html=True)
        st.markdown('<div class="charm">✨</div>', unsafe_allow_html=True)
        st.markdown('<div class="charm">🖌️</div>', unsafe_allow_html=True)
        st.markdown('<div class="charm">💎</div>', unsafe_allow_html=True)
        st.markdown('<div class="charm">🌈</div>', unsafe_allow_html=True)

    # 2. Main Workspace
    with col_main:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
            content_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="c")
            if content_file:
                content_img = Image.open(content_file).convert("RGB")
                st.image(content_img, use_column_width=True)

        with m_col2:
            st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
            style_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="s")
            if style_file:
                style_img = Image.open(style_file).convert("RGB")
                st.image(style_img, use_column_width=True)

        if content_file and style_file:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ CREATE MASTERPIECE"):
                with st.status("🎨 Applying High-Intensity Style..."):
                    model = get_model()
                    c_tensor = preprocess(content_img)
                    s_tensor = preprocess(style_img)
                    
                    s_out = model(s_tensor)
                    s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                    c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                    gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                    opt = tf.optimizers.Adam(learning_rate=5.0)

                    # Hardcoded High Intensity Style
                    for i in range(151):
                        with tf.GradientTape() as tape:
                            outs = model(gen_img)
                            sl = tf.add_n([tf.reduce_mean((gram_matrix(outs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                            cl = tf.add_n([tf.reduce_mean((outs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                            loss = (1e-1 * sl) + (1e4 * cl) # 1e-1 is High Style Intensity
                        
                        grad = tape.gradient(loss, gen_img)
                        opt.apply_gradients([(grad, gen_img)])
                
                final_art = deprocess(gen_img)
                st.markdown("<div style='text-align:center; padding:40px;'><h1>Final Masterpiece</h1></div>", unsafe_allow_html=True)
                st.image(final_art, use_column_width=True)

    # 3. Right Inspiration Gallery
    with col_gallery:
        st.markdown('<p style="color:#f1c40f; font-weight:bold; text-align:center;">STYLIZATION TIPS</p>', unsafe_allow_html=True)
        
        gallery_items = [
            ("Van Gogh", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
            ("Abstract", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Modern_mosaic.jpg/300px-Modern_mosaic.jpg")
        ]
        
        for name, url in gallery_items:
            st.markdown(f'''
                <div class="gallery-card">
                    <img src="{url}" width="100%" style="border-radius:5px;">
                    <div class="gallery-text">{name}</div>
                </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
