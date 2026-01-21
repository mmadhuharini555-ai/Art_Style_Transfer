import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 ADVANCED UI CONFIGURATION
# -------------------------------
def apply_advanced_theme():
    bg_img_url = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Radiant Glowing Title */
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 900;
            font-size: 90px !important;
            text-align: center;
            background: linear-gradient(to right, #00f2fe, #4facfe, #7117ea, #ea4444);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            filter: drop-shadow(0px 0px 20px rgba(79, 172, 254, 0.5));
        }}

        /* BIGGER LABELS FOR UPLOADERS */
        .big-label {{
            font-size: 45px !important;
            font-weight: 800 !important;
            color: #4facfe !important;
            text-align: center;
            margin-bottom: 20px;
            text-transform: uppercase;
            text-shadow: 2px 2px 15px rgba(0,0,0,0.5);
        }}

        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.05);
            border: 3px dashed rgba(79, 172, 254, 0.4);
            border-radius: 25px;
            padding: 30px;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: rgba(0, 0, 0, 0.8) !important;
            border-right: 1px solid #4facfe;
        }}

        .stButton>button {{
            width: 100%;
            background: linear-gradient(45deg, #7117ea, #ea4444);
            color: white;
            font-weight: bold;
            font-size: 1.5rem;
            padding: 20px;
            border-radius: 15px;
            border: none;
            transition: 0.3s;
        }}
        
        .stButton>button:hover {{
            box-shadow: 0 0 40px rgba(234, 68, 68, 0.6);
            transform: scale(1.02);
        }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE (VGG19)
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

def load_img_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

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
    st.set_page_config(page_title="Neural Art Studio Pro", layout="wide")
    apply_advanced_theme()

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:white; letter-spacing:3px;">THE AI-POWERED ARTISTIC REVOLUTION</p>', unsafe_allow_html=True)

    # --- SIDEBAR: Studio Controls ---
    st.sidebar.markdown("# 🎨 Studio Controls")
    st.sidebar.info("Fine-tune how the AI paints your image.")
    iterations = st.sidebar.slider("Brush Strokes (Quality)", 50, 500, 100)
    style_strength = st.sidebar.select_slider("Style Strength", options=["Subtle", "Medium", "Heavy"], value="Medium")
    
    weight_map = {"Subtle": 1e-4, "Medium": 1e-2, "Heavy": 1e-1}
    s_weight = weight_map[style_strength]

    # --- STYLE PRESETS ---
    st.markdown("### 🌟 Quick Presets (Click to choose a style)")
    presets = {
        "Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "The Scream": "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
        "Mosaic": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Modern_mosaic.jpg/1200px-Modern_mosaic.jpg",
        "Picasso": "https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Picasso_Grand_Nu_au_Fauteuil_Rouge.jpg/220px-Picasso_Grand_Nu_au_Fauteuil_Rouge.jpg"
    }
    
    selected_preset = st.selectbox("Or select a famous masterpiece style:", ["None"] + list(presets.keys()))

    # --- WORKSPACE ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        content_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="content")
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            st.image(content_img, use_column_width=True)

    with col2:
        st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
        style_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="style")
        
        # Logic for Style (Upload or Preset)
        final_style_img = None
        if style_file:
            final_style_img = Image.open(style_file).convert("RGB")
            st.image(final_style_img, use_column_width=True)
        elif selected_preset != "None":
            final_style_img = load_img_from_url(presets[selected_preset])
            st.image(final_style_img, caption=f"Preset: {selected_preset}", use_column_width=True)

    # --- GENERATION ---
    if content_file and final_style_img:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("✨ START ARTISTIC TRANSFORMATION"):
            with st.status("🔮 AI is analyzing textures and colors...", expanded=True) as status:
                model = get_model()
                c_tensor = preprocess(content_img)
                s_tensor = preprocess(final_style_img)

                s_out = model(s_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                opt = tf.optimizers.Adam(learning_rate=5.0)

                for i in range(iterations + 1):
                    with tf.GradientTape() as tape:
                        outputs = model(gen_img)
                        sl = tf.add_n([tf.reduce_mean((gram_matrix(outputs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                        cl = tf.add_n([tf.reduce_mean((outputs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                        loss = (s_weight * sl) + (1e4 * cl)
                    
                    grad = tape.gradient(loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                    if i % 10 == 0:
                        st.write(f"Refining brushstrokes: {i}/{iterations}")
                
                status.update(label="✅ Transformation Complete!", state="complete")

            final_art = deprocess(gen_img)
            
            # Show Result
            st.markdown("<div style='text-align:center; padding:40px;'><h1>Your Masterpiece</h1></div>", unsafe_allow_html=True)
            st.image(final_art, use_column_width=True)
            
            # Download Button
            res_pil = Image.fromarray(final_art)
            res_pil.save("masterpiece.png")
            with open("masterpiece.png", "rb") as f:
                st.download_button("📥 SAVE TO GALLERY", f, "neural_artwork.png", "image/png")

if __name__ == "__main__":
    main()
