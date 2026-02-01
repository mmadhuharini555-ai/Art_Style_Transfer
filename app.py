import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

# -------------------------------
# 🎨 UI & SAKURA STYLING
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
        
        /* SIDEBAR NAMES - HIGH VISIBILITY VIBRANT CYAN */
        [data-testid="stSidebar"] [data-testid="stImageCaption"] {{
            color: #00f2ff !important;
            font-weight: 800 !important;
            font-size: 1.0rem !important;
            text-shadow: 1px 1px 5px rgba(0,0,0,0.9);
            text-align: center;
        }}

        /* Expander Headers Styling */
        .streamlit-expanderHeader {{
            color: #ee0979 !important;
            font-weight: 800 !important;
            font-size: 1.2rem !important;
            background: rgba(255, 255, 255, 0.05) !important;
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
        }}
        .big-label {{ font-size: 32px !important; font-weight: 800; color: #00c6ff; text-align: center; }}
        [data-testid="stSidebar"] {{ background: rgba(0, 0, 0, 0.85) !important; border-right: 2px solid #ee0979; }}
    </style>
    
    <div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
    <div class="sakura" style="left:25%; animation-duration:15s; animation-delay:2s;">🌸</div>
    <div class="sakura" style="left:45%; animation-duration:12s; animation-delay:4s;">🌸</div>
    <div class="sakura" style="left:65%; animation-duration:18s; animation-delay:1s;">🌸</div>
    <div class="sakura" style="left:85%; animation-duration:14s; animation-delay:6s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE (KEEPING YOUR SPEED)
# -------------------------------
@st.cache_resource
def load_fast_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_and_prep_img(image_file, target_dim=512):
    img = Image.open(image_file).convert('RGB')
    img.thumbnail((target_dim, target_dim))
    img_array = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.constant(img_array), img

# -------------------------------
# 🚀 APP INTERFACE
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_ui_design()

    if 'history' not in st.session_state:
        st.session_state.history = []

    # --- 🏰 SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("<h1 style='color:#ee0979; text-align:center;'>✨ Studio Menu</h1>", unsafe_allow_html=True)
        
        # 🎨 GALLERY EXPANDER
        with st.expander("🎨 Gallery Inspirations 💎", expanded=False):
            styles_ref = [
                ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
                ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
                ("Mosaic Art", "https://images.unsplash.com/photo-1543857778-c4a1a3e0b2eb?w=300"),
                ("Oil Abstraction", "https://images.unsplash.com/photo-1549490349-8643362247b5?w=300")
            ]
            for name, url in styles_ref:
                st.image(url, caption=name, use_column_width=True)

        # 🏺 HISTORY VAULT EXPANDER
        with st.expander("🏺 History Vault 📜", expanded=False):
            if not st.session_state.history:
                st.info("Your creations will appear here.")
            else:
                for idx, item in enumerate(st.session_state.history):
                    st.image(item['masterpiece'], caption=f"Creation #{len(st.session_state.history) - idx}", use_column_width=True)
                    hist_buf = io.BytesIO()
                    item['masterpiece'].save(hist_buf, format="PNG")
                    st.download_button(
                        label=f"Download #{len(st.session_state.history) - idx}", 
                        data=hist_buf.getvalue(), 
                        file_name=f"history_{idx}.png", 
                        mime="image/png",
                        key=f"sidebar_btn_{idx}",
                        use_container_width=True
                    )
                    st.markdown("<hr style='border: 0.5px solid #00c6ff;'>", unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Alchemy of Styles</h1>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("Upload Content", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if c_file: st.image(c_file, use_column_width=True)

    with c2:
        st.markdown('<p class="big-label">🎨 Artistic Style</p>', unsafe_allow_html=True)
        s_file = st.file_uploader("Upload Style", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if s_file: st.image(s_file, use_column_width=True)

    if c_file and s_file:
        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🌸 Creating Masterpiece...", expanded=True):
                model = load_fast_model()
                content_tensor, content_pil = load_and_prep_img(c_file, target_dim=1024)
                style_tensor, _ = load_and_prep_img(s_file, target_dim=512)
                
                outputs = model(content_tensor, style_tensor)
                final_art_np = np.array(outputs[0][0] * 255).astype(np.uint8)
                final_art_pil = Image.fromarray(final_art_np)
                
                st.session_state.history.insert(0, {
                    "masterpiece": final_art_pil,
                    "original": content_pil
                })
            
            st.markdown("""
                <div style='text-align:center; margin-top: 20px; margin-bottom: 20px;'>
                    <h2 style='color: #fff; text-shadow: 0 0 10px #ff00de, 0 0 20px #ff00de, 0 0 40px #ff00de; font-weight: 900; font-size: 45px;'>
                        🌸 TRANSFORMATION COMPLETE! 🌸
                    </h2>
                </div>
            """, unsafe_allow_html=True)

            comp1, comp2 = st.columns(2)
            with comp1:
                st.markdown("<h3 style='color:#00c6ff; text-align:center;'>Before</h3>", unsafe_allow_html=True)
                st.image(content_pil, use_column_width=True)
            with comp2:
                st.markdown("<h3 style='color:#ee0979; text-align:center;'>After</h3>", unsafe_allow_html=True)
                st.image(final_art_pil, use_column_width=True)
            
            buf = io.BytesIO()
            final_art_pil.save(buf, format="PNG")
            st.download_button(label="📥 DOWNLOAD THIS MASTERPIECE", data=buf.getvalue(), file_name="masterpiece.png", mime="image/png", use_container_width=True)

if __name__ == "__main__":
    main()
