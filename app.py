import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🎨 UI & SAKURA STYLING
# -------------------------------
def apply_ui_design():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=1920"
    
    st.markdown(f"""
    <style>
        /* Lightened Background */
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
        }}

        /* Radiant Blue-to-Pink Title */
        .main-title {{
            font-weight: 900;
            font-size: 70px !important;
            text-align: center;
            background: linear-gradient(to right, #00c6ff, #f781f3, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            filter: drop-shadow(0 0 10px rgba(0, 198, 255, 0.5));
        }}

        /* Sakura Animation */
        .sakura {{
            position: fixed; top: -10%; z-index: 999;
            color: #ffb7c5; font-size: 25px;
            animation: fall linear infinite;
            pointer-events: none;
        }}
        @keyframes fall {{
            0% {{ transform: translateY(0vh) rotate(0deg) translateX(0px); opacity: 1; }}
            100% {{ transform: translateY(110vh) rotate(360deg) translateX(100px); opacity: 0; }}
        }}

        /* UPLOADER TEXT VISIBILITY FIX */
        [data-testid="stFileUploadDropzone"] div {{
            color: #000000 !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stFileUploadDropzone"] button span {{
            color: #000000 !important;
            font-weight: 800 !important;
        }}

        [data-testid="stFileUploadDropzone"] small {{
            color: #333333 !important;
            font-weight: 600 !important;
        }}

        /* Radiant Paint Masterpiece Button */
        .stButton>button {{
            background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
            color: white !important;
            font-weight: 800 !important;
            border-radius: 12px !important;
            border: none !important;
            height: 55px;
            width: 100%;
        }}

        /* Radiant Download Button */
        .stDownloadButton>button {{
            background: linear-gradient(45deg, #f781f3, #ee0979) !important;
            color: white !important;
            font-weight: 800 !important;
            border-radius: 12px !important;
            border: none !important;
            margin-top: 10px;
        }}

        .big-label {{
            font-size: 32px !important;
            font-weight: 800;
            color: #00c6ff;
            text-align: center;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(0, 0, 0, 0.7) !important;
            border-right: 2px solid #ee0979;
        }}
    </style>

    <div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
    <div class="sakura" style="left:25%; animation-duration:15s; animation-delay:2s;">🌸</div>
    <div class="sakura" style="left:45%; animation-duration:12s; animation-delay:4s;">🌸</div>
    <div class="sakura" style="left:65%; animation-duration:18s; animation-delay:1s;">🌸</div>
    <div class="sakura" style="left:85%; animation-duration:14s; animation-delay:6s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE (ULTRA SPEED)
# -------------------------------
IMG_SIZE = 450 

@st.cache_resource
def get_vgg_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    return Model(vgg.input, [vgg.get_layer(n).output for n in style_layers + content_layers])

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    return result / tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)

@tf.function()
def train_step(gen_img, s_targets, c_targets, model, opt):
    with tf.GradientTape() as tape:
        outputs = model(gen_img)
        sl = tf.add_n([tf.reduce_mean((gram_matrix(outputs[i]) - s_targets[i])**2) for i in range(5)])
        cl = tf.add_n([tf.reduce_mean((outputs[5+i] - c_targets[i])**2) for i in range(1)])
        loss = (1e1 * sl) + (1e4 * cl) 
    grad = tape.gradient(loss, gen_img)
    opt.apply_gradients([(grad, gen_img)])
    gen_img.assign(tf.clip_by_value(gen_img, -128, 128))

# -------------------------------
# 🚀 APP INTERFACE
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_ui_design()

    with st.sidebar:
        st.markdown("<h1 style='color:#ee0979;'>Gallery Inspirations</h1>", unsafe_allow_html=True)
        styles_ref = [
            ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
            ("Mosaic Art", "https://images.unsplash.com/photo-1543857778-c4a1a3e0b2eb?w=300"),
            ("Oil Abstraction", "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=300")
        ]
        for name, url in styles_ref:
            st.image(url, caption=name, use_column_width=True)

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("C", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if c_file: st.image(Image.open(c_file), use_column_width=True)

    with c2:
        st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("S", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if s_file: st.image(Image.open(s_file), use_column_width=True)

    if c_file and s_file:
        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🌸 Creating masterpiece..."):
                model = get_vgg_model()
                c_img = Image.open(c_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                s_img = Image.open(s_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                
                c_arr = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img_to_array(c_img), 0))
                s_arr = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img_to_array(s_img), 0))

                s_targets = [gram_matrix(out) for out in model(s_arr)[:5]]
                c_targets = model(c_arr)[5:]

                gen_img = tf.Variable(c_arr, dtype=tf.float32)
                opt = tf.optimizers.Adam(learning_rate=20.0)

                for _ in range(25): 
                    train_step(gen_img, s_targets, c_targets, model, opt)
            
            img = gen_img.numpy().reshape((IMG_SIZE, IMG_SIZE, 3))
            img[:, :, 0] += 103.939; img[:, :, 1] += 116.779; img[:, :, 2] += 123.68
            final_art = np.clip(img[:, :, ::-1], 0, 255).astype('uint8')
            
            st.markdown("<div style='text-align:center;'><h2>Transformation Complete!</h2></div>", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns([1, 1, 1])
            with m2:
                st.image(final_art, width=380)
                buf = io.BytesIO()
                Image.fromarray(final_art).save(buf, format="PNG")
                st.download_button(label="📥 DOWNLOAD MASTERPIECE", data=buf.getvalue(), file_name="masterpiece.png", mime="image/png")

if __name__ == "__main__":
    main()
