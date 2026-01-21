import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
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
            background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Radiant Title Styling */
        .main-title {{
            font-family: 'Montserrat', sans-serif;
            font-weight: 900;
            font-size: 85px !important;
            text-align: center;
            /* Radiant Gradient Effect */
            background: linear-gradient(to right, #00f2fe 0%, #4facfe 30%, #00f2fe 60%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            filter: drop-shadow(0px 0px 15px rgba(79, 172, 254, 0.8));
            letter-spacing: -2px;
        }}

        .sub-title {{
            font-family: 'Montserrat', sans-serif;
            text-align: center;
            letter-spacing: 5px;
            color: #ffffff;
            font-size: 16px;
            font-weight: 300;
            margin-bottom: 50px;
            text-transform: uppercase;
            opacity: 0.9;
        }}

        /* File Uploader Container */
        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 2px dashed rgba(79, 172, 254, 0.5);
            padding: 25px;
        }}

        /* Labels for Uploaders */
        [data-testid="stFileUploader"] label {{
            color: #4facfe !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 20px !important;
            text-shadow: 0px 0px 10px rgba(79, 172, 254, 0.3);
        }}

        /* Make "Drag and drop" text clearly visible */
        [data-testid="stFileUploadDropzone"] div {{
            color: #ffffff !important;
        }}

        /* Button Styling */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(45deg, #4facfe, #00f2fe);
            color: white;
            font-weight: bold;
            border: none;
            padding: 18px;
            border-radius: 12px;
            font-size: 1.2rem;
            transition: 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
        }}

        .stButton>button:hover {{
            box-shadow: 0 0 30px rgba(79, 172, 254, 0.6);
            transform: translateY(-2px);
        }}

        /* Clean up Streamlit UI */
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
    img = img_to_array(img)
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

    col1, col2 = st.columns(2)

    with col1:
        # Reverted back to Subject Image
        content_file = st.file_uploader("🖼️ Subject Image", type=["jpg", "png", "jpeg"])
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            st.image(content_img, use_column_width=True)

    with col2:
        # Reverted back to Artistic Style
        style_file = st.file_uploader("🎨 Artistic Style", type=["jpg", "png", "jpeg"])
        if style_file:
            style_img = Image.open(style_file).convert("RGB")
            st.image(style_img, use_column_width=True)

    if content_file and style_file:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🖌️ Generate Masterpiece"):
            with st.status("🎨 Mixing colors and applying AI brushstrokes...", expanded=True):
                model = get_model()
                c_tensor = load_and_process_image(content_img)
                s_tensor = load_and_process_image(style_img)

                s_out = model(s_tensor)
                s_feats = [gram_matrix(out) for out in s_out[:len(STYLE_LAYERS)]]
                c_feats = model(c_tensor)[len(STYLE_LAYERS):]

                gen_img = tf.Variable(c_tensor, dtype=tf.float32)
                opt = tf.optimizers.Adam(learning_rate=5.0)

                for i in range(101):
                    with tf.GradientTape() as tape:
                        outputs = model(gen_img)
                        s_loss = tf.add_n([tf.reduce_mean((gram_matrix(outputs[i]) - s_feats[i])**2) for i in range(len(STYLE_LAYERS))])
                        c_loss = tf.add_n([tf.reduce_mean((outputs[len(STYLE_LAYERS)+i] - c_feats[i])**2) for i in range(len(CONTENT_LAYERS))])
                        total_loss = (1e-2 * s_loss) + (1e4 * c_loss)
                    
                    grad = tape.gradient(total_loss, gen_img)
                    opt.apply_gradients([(grad, gen_img)])
                
            final_image = deprocess_image(gen_img)
            st.markdown("<div style='text-align:center; padding-top:40px;'><h1>Final Masterpiece</h1></div>", unsafe_allow_html=True)
            st.image(final_image, use_column_width=True)

if __name__ == "__main__":
    main()
