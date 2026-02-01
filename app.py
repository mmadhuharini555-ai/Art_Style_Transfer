import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
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
            font-weight: 900; font-size: 70px !important; text-align: center;
            background: linear-gradient(to right, #00c6ff, #f781f3, #ee0979);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 20px; filter: drop-shadow(0 0 10px rgba(0, 198, 255, 0.5));
        }}
        [data-testid="stSidebar"] [data-testid="stImageCaption"] {{
            color: #00f2ff !important; font-weight: 800 !important; font-size: 1.0rem !important;
            text-shadow: 1px 1px 5px rgba(0,0,0,0.9); text-align: center;
        }}
        .sakura {{
            position: fixed; top: -10%; z-index: 999; color: #ffb7c5; font-size: 25px;
            animation: fall linear infinite; pointer-events: none;
        }}
        @keyframes fall {{
            0% {{ transform: translateY(0vh) rotate(0deg) translateX(0px); opacity: 1; }}
            100% {{ transform: translateY(110vh) rotate(360deg) translateX(100px); opacity: 0; }}
        }}
        .stButton>button {{
            background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
            color: white !important; font-weight: 800 !important;
            border-radius: 12px !important; border: none !important; height: 55px; width: 100%;
        }}
        .stDownloadButton>button {{
            background: linear-gradient(45deg, #f781f3, #ee0979) !important;
            color: white !important; font-weight: 800 !important;
            border-radius: 12px !important; border: none !important;
        }}
        .big-label {{ font-size: 32px !important; font-weight: 800; color: #00c6ff; text-align: center; }}
        [data-testid="stSidebar"] {{ background: rgba(0, 0, 0, 0.85) !important; border-right: 2px solid #ee0979; }}
        .control-panel {{
            background: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 15px;
            border: 1px solid #ee0979; margin-top: 20px;
        }}
    </style>
    <div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
    <div class="sakura" style="left:25%; animation-duration:15s; animation-delay:2s;">🌸</div>
    <div class="sakura" style="left:45%; animation-duration:12s; animation-delay:4s;">🌸</div>
    <div class="sakura" style="left:65%; animation-duration:18s; animation-delay:1s;">🌸</div>
    <div class="sakura" style="left:85%; animation-duration:14s; animation-delay:6s;">🌸</div>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 AI ENGINE
# -------------------------------
@st.cache_resource
def load_fast_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def prep_img_for_model(img, target_dim):
    img = img.resize((target_dim, target_dim))
    img_array = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.constant(img_array)

# -------------------------------
# 🚀 APP INTERFACE
# -------------------------------
def main():
    st.set_page_config(page_title="Neural Art Pro", layout="wide")
    apply_ui_design()

    if 'history' not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown("<h1 style='color:#ee0979; text-align:center;'>✨ Studio Menu</h1>", unsafe_allow_html=True)
        
        with st.expander("🎨 Gallery Inspirations 💎", expanded=False):
            styles_ref = [
                ("Starry Night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
                ("The Scream", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/300px-The_Scream.jpg"),
                ("Mosaic Art", "https://images.unsplash.com/photo-1543857778-c4a1a3e0b2eb?w=300"),
                ("Oil Abstraction", "https://images.unsplash.com/photo-1549490349-8643362247b5?w=300")
            ]
            for name, url in styles_ref:
                st.image(url, caption=name, use_column_width=True)

        with st.expander("🏺 History Vault 📜", expanded=False):
            if not st.session_state.history:
                st.info("Vault is empty...")
            else:
                for idx, item in enumerate(st.session_state.history):
                    st.image(item['masterpiece'], caption=f"Creation #{len(st.session_state.history) - idx}", use_column_width=True)
                    st.markdown("<hr style='border: 0.5px solid #00c6ff;'>", unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Alchemy of Styles</h1>', unsafe_allow_html=True)

    # --- 📸 INPUT SECTION ---
    c_col, s_col1, s_col2 = st.columns([2, 1, 1])
    with c_col:
        st.markdown('<p class="big-label">🖼️ Subject Image</p>', unsafe_allow_html=True)
        c_file = st.file_uploader("C", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    with s_col1:
        st.markdown('<p class="big-label" style="font-size:22px !important;">🎨 Artistic Style A</p>', unsafe_allow_html=True)
        s_file1 = st.file_uploader("S1", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    with s_col2:
        st.markdown('<p class="big-label" style="font-size:22px !important;">🎨 Artistic Style B</p>', unsafe_allow_html=True)
        s_file2 = st.file_uploader("S2", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if c_file and (s_file1 or s_file2):
        # --- 🛠️ ADVANCED SETTINGS PANEL ---
        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["🎚️ Style Control", "🌓 Darkroom", "✒️ Artist Pro"])
        
        with t1:
            col_a, col_b = st.columns(2)
            with col_a:
                strength = st.slider("✨ Alchemy Strength", 0.0, 1.0, 0.8)
                res_mode = st.radio("📐 Resolution", ["Fast Draft (512px)", "Gallery Print (1024px)"], horizontal=True)
            with col_b:
                fusion = st.slider("🔗 Style Fusion (A vs B)", 0.0, 1.0, 0.5) if (s_file1 and s_file2) else st.write("Upload two styles to fuse.")
        
        with t2:
            d_col1, d_col2, d_col3 = st.columns(3)
            bright = d_col1.slider("☀️ Brightness", 0.5, 2.0, 1.0)
            contr = d_col2.slider("🌓 Contrast", 0.5, 2.0, 1.0)
            sharp = d_col3.slider("🔪 Sharpness", 0.5, 3.0, 1.0)

        with t3:
            signature = st.text_input("🖋️ Add Artist Signature (Optional)", "")
            sig_color = st.color_picker("🎨 Signature Color", "#00f2ff")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("✨ PAINT MASTERPIECE"):
            with st.status("🌸 Transmuting Art...", expanded=True):
                model = load_fast_model()
                target_dim = 512 if "Fast" in res_mode else 1024
                
                # 1. PREP CONTENT (Darkroom)
                content_pil = Image.open(c_file).convert("RGB")
                content_pil = ImageEnhance.Brightness(content_pil).enhance(bright)
                content_pil = ImageEnhance.Contrast(content_pil).enhance(contr)
                content_pil = ImageEnhance.Sharpness(content_pil).enhance(sharp)
                content_tensor = prep_img_for_model(content_pil, target_dim)

                # 2. PREP STYLE (Fusion)
                if s_file1 and s_file2:
                    s1 = Image.open(s_file1).convert("RGB").resize((512, 512))
                    s2 = Image.open(s_file2).convert("RGB").resize((512, 512))
                    style_pil = Image.blend(s1, s2, fusion)
                else:
                    style_pil = Image.open(s_file1 if s_file1 else s_file2).convert("RGB")
                style_tensor = prep_img_for_model(style_pil, 512)

                # 3. RUN AI
                outputs = model(content_tensor, style_tensor)
                stylized_np = np.array(outputs[0][0] * 255).astype(np.uint8)
                stylized_pil = Image.fromarray(stylized_np)

                # 4. ALCHEMY BLEND (Strength)
                final_art = Image.blend(content_pil.resize(stylized_pil.size), stylized_pil, strength)

                # 5. ADD SIGNATURE
                if signature:
                    draw = ImageDraw.Draw(final_art)
                    # Basic font logic
                    draw.text((20, final_art.size[1]-50), signature, fill=sig_color)

                st.session_state.history.insert(0, {"masterpiece": final_art})
            
            st.markdown("<div style='text-align:center;'><h2 style='color: #fff; text-shadow: 0 0 20px #ff00de; font-weight: 900; font-size: 45px;'>🌸 TRANSFORMATION COMPLETE! 🌸</h2></div>", unsafe_allow_html=True)

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.image(content_pil, caption="Original Subject", use_column_width=True)
            with m_col2:
                st.image(final_art, caption="Stylized Masterpiece", use_column_width=True)
                buf = io.BytesIO()
                final_art.save(buf, format="PNG")
                st.download_button("📥 DOWNLOAD MASTERPIECE", buf.getvalue(), "art.png", "image/png", use_container_width=True)

if __name__ == "__main__":
    main()

