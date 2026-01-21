import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 DYNAMIC ART STUDIO UI
# -------------------------------
def apply_dynamic_theme():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.88)), url("{bg_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Radiant Title */
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 900;
            font-size: 70px !important;
            text-align: center;
            background: linear-gradient(45deg, #00f2fe, #4facfe, #f1c40f, #ff0080);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }}

        /* Left Inspiration Gallery */
        .inspiration-box {{
            border-left: 3px solid #4facfe;
            padding-left: 15px;
            margin-bottom: 25px;
        }}
        .insp-img {{
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: 0.3s;
        }}
        .insp-img:hover {{ transform: scale(1.05); border-color: #f1c40f; }}

        /* Smaller Floating Gradient Charms */
        .charm {{
            position: fixed;
            font-size: 25px;
            z-index: -1;
            background: linear-gradient(45deg, #f1c40f, #ff0080);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: move 10s linear infinite;
            opacity: 0.6;
        }}

        @keyframes move {{
            0% {{ transform: translate(0, 0) rotate(0deg); }}
            25% {{ transform: translate(100px, 200px) rotate(90deg); }}
            50% {{ transform: translate(200px, 50px) rotate(180deg); }}
            75% {{ transform: translate(-50px, 150px) rotate(270deg); }}
            100% {{ transform: translate(0, 0) rotate(360deg); }}
        }}

        /* Workspace Styling */
        .big-label {{
            font-size: 30px !important;
            font-weight: 800;
            color: #4facfe;
            text-align: center;
            margin-bottom: 10px;
        }}

        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px dashed #4facfe;
            border-radius: 15px;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
    </style>

    <!-- Floating Charms -->
    <div class="charm" style="top:10%; left:5%; animation-duration: 15s;">🎨</div>
    <div class="charm" style="top:40%; left:85%; animation-duration: 20s;">✨</div>
    <div class="charm" style="top:70%; left:15%; animation-duration: 12s;">🖌️</div>
    <div class="charm" style="top:20%; left:70%; animation-duration: 18s;">🌈</div>
    <div class="charm" style="top:80%; left:60%; animation-duration: 25s;">💎</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 OPTIMIZED AI ENGINE
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
# 🚀 MAIN APPLICATION
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_dynamic_theme()

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    # LAYOUT: Inspirations (Left) | Main Workspace (Right)
    col_insp, col_work = st.columns([1.5, 6])

    with col_insp:
        st.markdown('<p style="color:#f1c40f; font-weight:bold; font-size:18px;">INSPIRATIONS</p>', unsafe_allow_html=True)
        styles = [
            ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/200px-The_Scream.jpg"),
            ("Mosaic Art", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Modern_mosaic.jpg/200px-Modern_mosaic.jpg"),
            ("Oil Abstract", "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=200")
        ]
        for name, url in styles:
            st.markdown(f'<div class="inspiration-box"><img src="{url}" class="insp-img"><br><small>{name}</small></div>', unsafe_allow_html=True)

    with col_work:
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.markdown('<p class="big-label">🖼️ Subject</p>', unsafe_allow_html=True)
            c_file = st.file_uploader("Upload content", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if c_file:
                c_img = Image.open(c_file).convert("RGB")
                st.image(c_img, use_column_width=True)

        with w_col2:
            st.markdown('<p class="big-label">🎨 Style</p>', unsafe_allow_html=True)
            s_file = st.file_uploader("Upload style", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if s_file:
                s_img = Image.open(s_file).convert("RGB")
                st.image(s_img, use_column_width=True)

        if c_file and s_file:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ PAINT MASTERPIECE"):
                with st.status("🔮 Transforming into Art... (Fast Mode)"):
                    model = get_model()
                    c_tensor = preprocess(c_img)
                    s_tensor = preprocess(s_img)
                    
                    s_out = model(s_tensor)
                    s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                    c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                    gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                    # Increased Learning Rate for faster convergence
                    opt = tf.optimizers.Adam(learning_rate=10.0)

                    # Faster loop: only 70 iterations
                    for i in range(71):
                        with tf.GradientTape() as tape:
                            outs = model(gen_img)
                            # Heavy Style Weight (1.0) vs Content Weight (1e4)
                            # This ensures the style is VERY visible
                            sl = tf.add_n([tf.reduce_mean((gram_matrix(outs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                            cl = tf.add_n([tf.reduce_mean((outs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                            loss = (1e0 * sl) + (1e4 * cl) 
                        
                        grad = tape.gradient(loss, gen_img)
                        opt.apply_gradients([(grad, gen_img)])
                
                final_art = deprocess(gen_img)
                st.markdown("<div style='text-align:center; padding-top:20px;'><p class='big-label' style='color:#f1c40f;'>MASTERPIECE COMPLETE</p></div>", unsafe_allow_html=True)
                
                # MINIMIZED IMAGE: Controlled width
                col_left, col_mid, col_right = st.columns([1, 2, 1])
                with col_mid:
                    st.image(final_art, width=400, caption="Minimized High-Intensity Output")
                    
if __name__ == "__main__":
    main()
