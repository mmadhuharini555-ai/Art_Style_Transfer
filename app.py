import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# -------------------------------
# 🌸 RADIANT UI & SAKURA EFFECTS
# -------------------------------
def apply_ui_design():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=1920"
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
        }}

        /* Radiant Blue-to-Pink Title */
        .main-title {{
            font-weight: 900;
            font-size: 75px !important;
            text-align: center;
            background: linear-gradient(45deg, #00c6ff, #f781f3, #ee0979);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }}

        /* BUTTONS & UPLOADER TEXT VISIBILITY FIX */
        /* This ensures "Browse files", "Paint", and "Download" are always visible */
        button p, .stButton>button, label, .stMarkdown p {{
            color: white !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}
        
        /* Specific Fix for Browse Files button text */
        [data-testid="stFileUploadDropzone"] button span {{
            color: black !important; /* Standard button text color for visibility */
            font-weight: bold !important;
        }}

        .stButton>button {{
            background: linear-gradient(45deg, #00c6ff, #ee0979) !important;
            border: none !important;
            padding: 15px 30px !important;
            border-radius: 12px !important;
            font-size: 1.2rem !important;
        }}

        /* Cherry Blossom Animation */
        .sakura {{
            position: fixed; top: -10%; z-index: 999;
            color: #ffb7c5; font-size: 25px;
            animation: fall linear infinite;
        }}
        @keyframes fall {{
            0% {{ transform: translateY(0vh) rotate(0deg); opacity: 1; }}
            100% {{ transform: translateY(110vh) rotate(360deg); opacity: 0; }}
        }}

        .big-label {{
            font-size: 32px !important;
            font-weight: 800;
            color: #00c6ff;
            text-align: center;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(10, 10, 10, 0.95) !important;
            border-right: 2px solid #ee0979;
        }}
    </style>

    <!-- Sakura Petals -->
    <div class="sakura" style="left:10%; animation-duration:12s;">🌸</div>
    <div class="sakura" style="left:30%; animation-duration:18s; animation-delay:2s;">🌸</div>
    <div class="sakura" style="left:55%; animation-duration:15s; animation-delay:5s;">🌸</div>
    <div class="sakura" style="left:80%; animation-duration:20s; animation-delay:1s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🚀 HIGH-SPEED AI ENGINE
# -------------------------------
# Quality: 450px ensures sharp details
IMG_SIZE = 450 

@st.cache_resource
def get_vgg_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return Model(vgg.input, outputs)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
    return result / num_locations

# COMPILATION: This @tf.function makes the generation up to 5x faster
@tf.function()
def train_step(generated_image, style_targets, content_targets, model, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        style_outputs = outputs[:5]
        content_outputs = outputs[5:]
        
        # Style Loss
        s_loss = tf.add_n([tf.reduce_mean((gram_matrix(style_outputs[i]) - style_targets[i])**2) for i in range(5)])
        # Content Loss
        c_loss = tf.add_n([tf.reduce_mean((content_outputs[i] - content_targets[i])**2) for i in range(1)])
        
        # Hyper-weights for "Instant" stylization
        total_loss = (1e-1 * s_loss) + (1e4 * c_loss)

    grad = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, -128, 128))

def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_ui_design()

    with st.sidebar:
        st.markdown("<h1 style='color:#ee0979;'>Gallery Inspirations</h1>", unsafe_allow_html=True)
        # Verified High-Res Links
        styles = [
            ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
            ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
            ("Mosaic Pattern", "https://images.unsplash.com/photo-1579546929518-9e396f3cc809?w=300"),
            ("Oil Abstraction", "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=300")
        ]
        for name, url in styles:
            st.image(url, caption=name, use_column_width=True)

    st.markdown('<h1 class="main-title">Neural Art Gallery</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("Upload content", type=["jpg", "png", "jpeg"], key="c", label_visibility="collapsed")
        if c_file:
            st.image(Image.open(c_file), use_column_width=True)

    with col2:
        st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("Upload style", type=["jpg", "png", "jpeg"], key="s", label_visibility="collapsed")
        if s_file:
            st.image(Image.open(s_file), use_column_width=True)

    if c_file and s_file:
        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🌸 Compiling High-Speed Graph..."):
                model = get_vgg_model()
                c_img = Image.open(c_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                s_img = Image.open(s_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                
                # Preprocessing
                c_arr = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img_to_array(c_img), 0))
                s_arr = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(img_to_array(s_img), 0))

                # Extract Targets
                s_out = model(s_arr)
                s_targets = [gram_matrix(out) for out in s_out[:5]]
                c_targets = model(c_arr)[5:]

                gen_img = tf.Variable(c_arr, dtype=tf.float32)
                optimizer = tf.optimizers.Adam(learning_rate=20.0)

                # FAST MODE: Only 20 iterations due to @tf.function and high LR
                for _ in range(21):
                    train_step(gen_img, s_targets, c_targets, model, optimizer)
            
            # Deprocess
            img = gen_img.numpy().reshape((IMG_SIZE, IMG_SIZE, 3))
            img[:, :, 0] += 103.939; img[:, :, 1] += 116.779; img[:, :, 2] += 123.68
            final_art = np.clip(img[:, :, ::-1], 0, 255).astype('uint8')
            
            st.markdown("<div style='text-align:center;'><h2>Art Complete!</h2></div>", unsafe_allow_html=True)
            
            r1, r2, r3 = st.columns([1, 1.2, 1])
            with r2:
                st.image(final_art, width=400)
                # Download
                buf = io.BytesIO()
                Image.fromarray(final_art).save(buf, format="PNG")
                st.download_button(label="📥 DOWNLOAD MASTERPIECE", data=buf.getvalue(), file_name="art.png", mime="image/png")

if __name__ == "__main__":
    main()
