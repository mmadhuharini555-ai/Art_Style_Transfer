import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub  # Required for high-speed quality
import numpy as np
from PIL import Image
import io

# -------------------------------
# 🎨 UI & SAKURA STYLING (UNCHANGED)
# -------------------------------
def apply_ui_design():
    bg_img = "https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=1920"
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{bg_img}");
            background-size: cover;
            background-attachment: fixed;
        }}
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
        [data-testid="stFileUploadDropzone"] div {{ color: #000000 !important; font-weight: 700 !important; }}
        [data-testid="stFileUploadDropzone"] button span {{ color: #000000 !important; font-weight: 800 !important; }}
        [data-testid="stFileUploadDropzone"] small {{ color: #333333 !important; font-weight: 600 !important; }}
        .stButton>button {{
            background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
            color: white !important; font-weight: 800 !important;
            border-radius: 12px !important; border: none !important;
            height: 55px; width: 100%;
        }}
        .stDownloadButton>button {{
            background: linear-gradient(45deg, #f781f3, #ee0979) !important;
            color: white !important; font-weight: 800 !important;
            border-radius: 12px !important; border: none !important;
            margin-top: 10px;
        }}
        .big-label {{ font-size: 32px !important; font-weight: 800; color: #00c6ff; text-align: center; }}
        [data-testid="stSidebar"] {{ background: rgba(0, 0, 0, 0.7) !important; border-right: 2px solid #ee0979; }}
    </style>
    <div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
    <div class="sakura" style="left:25%; animation-duration:15s; animation-delay:2s;">🌸</div>
    <div class="sakura" style="left:45%; animation-duration:12s; animation-delay:4s;">🌸</div>
    <div class="sakura" style="left:65%; animation-duration:18s; animation-delay:1s;">🌸</div>
    <div class="sakura" style="left:85%; animation-duration:14s; animation-delay:6s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE (ULTRA SPEED & HIGH QUALITY)
# -------------------------------
@st.cache_resource
def load_model():
    # This model is specifically designed for real-time high-quality style transfer
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
    return img

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
            ("Oil Abstraction", "https://images.unsplash.com/photo-1549490349-8643362247b5?w=300")
        ]
        for name, url in styles_ref:
            st.image(url, caption=name, use_column_width=True)

    st.markdown('<h1 class="main-title">Alchemy of Styles</h1>', unsafe_allow_html=True)

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
                # Load the high-speed model
                model = load_model()
                
                # Preprocess images
                content_img = preprocess_image(c_file)
                style_img = preprocess_image(s_file)
                
                # Stylize image (takes < 2 seconds)
                outputs = model(tf.constant(content_img), tf.constant(style_img))
                stylized_img = outputs[0]

                # Convert back to displayable image
                final_art = np.array(stylized_img[0] * 255).astype(np.uint8)
            
            st.markdown("<div style='text-align:center;'><h2>Transformation Complete!</h2></div>", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns([1, 1, 1])
            with m2:
                st.image(final_art, use_column_width=True)
                buf = io.BytesIO()
                Image.fromarray(final_art).save(buf, format="PNG")
                st.download_button(label="📥 DOWNLOAD MASTERPIECE", data=buf.getvalue(), file_name="masterpiece.png", mime="image/png")

if __name__ == "__main__":
    main()
