"""
INSTANT — Realistic Human Face Generator (GAN)
Streamlit app to load a trained GAN Generator model and generate synthetic faces.

Supports:
- TensorFlow / Keras models (.h5, .keras)
- PyTorch models (.pt, .pth)  -> expects a scripted/traced model OR a model class defined below

Run:
    streamlit run app.py
"""

import io
import zipfile
import numpy as np
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Instant • AI Face Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM THEME / CSS  (dark, neon-gradient, creative)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root{
    --bg-1:#0b0f1a;
    --bg-2:#111629;
    --accent-1:#7f5af0;
    --accent-2:#2cb1ff;
    --accent-3:#ff5ec4;
    --text-light:#e6e8ef;
    --card:#161b2e;
}

.stApp{
    background: radial-gradient(1200px 600px at 10% -10%, #1b1440 0%, transparent 60%),
                radial-gradient(1000px 500px at 110% 10%, #0d2b45 0%, transparent 55%),
                linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
    color: var(--text-light);
}

/* Header banner */
.hero{
    padding: 28px 34px;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(127,90,240,0.20), rgba(44,177,255,0.15), rgba(255,94,196,0.15));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}
.hero h1{
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, var(--accent-2), var(--accent-1), var(--accent-3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.hero p{
    color: #b7bdd6;
    margin-top: 6px;
    font-size: 0.98rem;
}
.badge{
    display:inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight:700;
    letter-spacing: .04em;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: white;
    margin-right: 6px;
}

/* Cards */
.card{
    background: var(--card);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0f1326 0%, #0a0e1c 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Buttons */
.stButton>button{
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
    letter-spacing: .02em;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 4px 14px rgba(127,90,240,0.35);
}
.stButton>button:hover{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(44,177,255,0.45);
}

/* Metric / image caption */
.gen-caption{
    text-align:center;
    color:#9aa3c7;
    font-size:0.78rem;
    margin-top: -8px;
}

hr{ border-color: rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">GAN</span><span class="badge">DEEP LEARNING</span><span class="badge">COMPUTER VISION</span>
        <h1>🧬 Realistic Human Face Generator</h1>
        <p>Upload your trained Generator model and synthesize brand-new human faces from random latent noise.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — MODEL & GENERATION CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model")
    framework = st.radio(
        "Framework",
        ["TensorFlow / Keras (.h5 / .keras)", "PyTorch (.pt / .pth — TorchScript)"],
        index=0,
    )
    model_file = st.file_uploader(
        "Upload trained Generator model",
        type=["h5", "keras", "pt", "pth"],
        help="Upload the Generator model you trained (not the Discriminator).",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Generation Settings")
    latent_dim = st.number_input("Latent vector size (z-dim)", min_value=16, max_value=1024, value=100, step=4)
    img_size = st.selectbox("Output image size (used if model has no fixed shape)", [64, 128, 256], index=0)
    num_images = st.slider("Number of faces to generate", 1, 25, 9)
    seed = st.number_input("Random seed (-1 = random each time)", value=-1, step=1)

    st.markdown("---")
    st.markdown("### 🧪 Extra Features")
    do_interpolate = st.checkbox("Latent-space interpolation (morph between 2 faces)", value=False)
    interp_steps = st.slider("Interpolation steps", 3, 12, 6, disabled=not do_interpolate)

    generate_btn = st.button("✨ Generate Faces", use_container_width=True)

# ----------------------------------------------------------------------------
# MODEL LOADING (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tf_model(file_bytes, filename):
    import tensorflow as tf
    import tempfile, os
    suffix = ".h5" if filename.endswith(".h5") else ".keras"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    model = tf.keras.models.load_model(tmp_path, compile=False)
    os.unlink(tmp_path)
    return model


@st.cache_resource(show_spinner=False)
def load_torch_model(file_bytes, filename):
    import torch
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    model = torch.jit.load(tmp_path, map_location="cpu")
    model.eval()
    os.unlink(tmp_path)
    return model


def tensor_to_uint8_images(arr):
    """Normalize a batch of generator outputs to uint8 HWC images, robust to [-1,1] or [0,1] ranges."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        # channels-first (N,C,H,W) -> (N,H,W,C)
        arr = np.transpose(arr, (0, 2, 3, 1))
    lo, hi = arr.min(), arr.max()
    if lo < -0.01:  # roughly [-1, 1]
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def generate_from_tf(model, z):
    out = model.predict(z, verbose=0)
    return tensor_to_uint8_images(out)


def generate_from_torch(model, z):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(z).float()
        out = model(t).cpu().numpy()
    return tensor_to_uint8_images(out)


def make_latents(n, dim, rng):
    return rng.normal(0, 1, size=(n, dim)).astype(np.float32)


def images_to_zip_bytes(images):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, img in enumerate(images):
            im = Image.fromarray(img)
            b = io.BytesIO()
            im.save(b, format="PNG")
            zf.writestr(f"face_{i+1:03d}.png", b.getvalue())
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------------
# MAIN AREA
# ----------------------------------------------------------------------------
col_info, = st.columns(1)
with col_info:
    st.markdown(
        """
        <div class="card">
        <b>How it works:</b> upload the <i>Generator</i> half of your trained GAN. The app samples random
        latent vectors <code>z ~ N(0, 1)</code> and feeds them through your Generator to synthesize new faces —
        nothing here is a real photo.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if generate_btn:
    if model_file is None:
        st.error("⚠️ من فضلك ارفع ملف الـ Generator model الأول من السايدبار.")
    else:
        rng = np.random.default_rng(None if seed == -1 else int(seed))
        file_bytes = model_file.getvalue()
        is_tf = framework.startswith("TensorFlow")

        with st.spinner("Loading model & generating faces..."):
            try:
                if is_tf:
                    model = load_tf_model(file_bytes, model_file.name)
                else:
                    model = load_torch_model(file_bytes, model_file.name)
            except Exception as e:
                st.error(f"❌ Couldn't load the model: {e}")
                st.stop()

            try:
                z = make_latents(num_images, latent_dim, rng)
                images = generate_from_tf(model, z) if is_tf else generate_from_torch(model, z)
            except Exception as e:
                st.error(
                    f"❌ Generation failed: {e}\n\n"
                    "Check that the latent dim matches your model's input, and that the model "
                    "outputs an image tensor (N,H,W,C) or (N,C,H,W)."
                )
                st.stop()

        st.success(f"✅ Generated {len(images)} synthetic face(s)!")

        # Gallery grid
        cols_per_row = 5 if num_images > 5 else num_images
        rows = (len(images) + cols_per_row - 1) // cols_per_row
        idx = 0
        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for c in cols:
                if idx >= len(images):
                    break
                c.image(images[idx], use_container_width=True)
                c.markdown(f"<div class='gen-caption'>face_{idx+1:03d}</div>", unsafe_allow_html=True)
                idx += 1

        # Download all as zip
        zip_buf = images_to_zip_bytes(images)
        st.download_button(
            "⬇️ Download all faces (.zip)",
            data=zip_buf,
            file_name="generated_faces.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # Optional latent-space interpolation / morph
        if do_interpolate:
            st.markdown("### 🌀 Latent-Space Morph")
            z1 = make_latents(1, latent_dim, rng)[0]
            z2 = make_latents(1, latent_dim, rng)[0]
            alphas = np.linspace(0, 1, interp_steps)
            z_interp = np.stack([(1 - a) * z1 + a * z2 for a in alphas]).astype(np.float32)
            try:
                morph_images = generate_from_tf(model, z_interp) if is_tf else generate_from_torch(model, z_interp)
                mcols = st.columns(interp_steps)
                for i, mc in enumerate(mcols):
                    mc.image(morph_images[i], use_container_width=True)
                    mc.markdown(f"<div class='gen-caption'>α={alphas[i]:.2f}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Interpolation preview failed: {e}")
else:
    st.info("👈 ارفع الـ Generator model وحدد الإعدادات، وبعدين دوس **Generate Faces**.")

st.markdown("---")
st.caption("Built with Streamlit • Instant AI Bootcamp — Realistic Human Face Generation using GAN")
