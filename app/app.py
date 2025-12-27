
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from utils import (
    load_config, load_model, preprocess_pil, predict_probs,
    GradCAM, effnet_target_layer, tensor_to_display_gray
)

st.set_page_config(page_title="Neonatal MRI Age Bin Classifier", layout="centered")

st.title("Neonatal Brain MRI Age Bin Classifier")
st.warning("Educational demo only — NOT for clinical diagnosis or medical decision-making.")

st.markdown(
    "Upload a **single MRI slice image** (PNG/JPG). The model predicts a **developmental age bin** "
    "and shows a **Grad-CAM heatmap** for interpretability."
)

BASE = Path(__file__).parent
CFG_PATH = BASE / "train_config.json"
CKPT_PATH = BASE / "best_efficientnet_b0.pt"

def pretty_age_bin(code: str) -> str:
    code = str(code).strip().lower()
    if code.endswith("m"):
        n = int(code[:-1])
        return f"{n} month" + ("" if n == 1 else "s")
    if code.endswith("d") and "_" in code:
        a, b = code[:-1].split("_", 1)
        return f"{int(a)}–{int(b)} days"
    return code

@st.cache_resource
def load_assets():
    cfg = load_config(CFG_PATH)
    classes = cfg["classes_14"]
    model = load_model(CKPT_PATH, num_classes=len(classes))
    cam = GradCAM(model, effnet_target_layer(model))
    return classes, model, cam

classes, model, cam = load_assets()

# ---- Sidebar settings ----
st.sidebar.header("Settings")
show_cam = st.sidebar.checkbox("Show Grad-CAM", value=True)
topk = st.sidebar.slider("Top-K probabilities", 3, 8, 5)

# ---- Upload only ----
file = st.file_uploader("Upload MRI slice (png/jpg)", type=["png","jpg","jpeg"])
if not file:
    st.info("Upload an image to get a prediction.")
    st.stop()

img = Image.open(file).convert("RGB")
st.image(img, caption="Uploaded image", use_container_width=True)

# ---- Predict ----
x = preprocess_pil(img)
probs = predict_probs(model, x)

pred_idx = int(np.argmax(probs))
pred_label = pretty_age_bin(classes[pred_idx])
pred_conf = float(probs[pred_idx])

st.subheader("Prediction")
st.write(f"**Predicted age bin:** `{pred_label}`")
st.write(f"**Confidence:** `{pred_conf:.3f}`")

# Extra disclaimer under results
st.caption("Educational demo only. Outputs may be wrong. Do not use for medical decisions.")

k = min(topk, len(classes))
top_idx = np.argsort(probs)[::-1][:k]
st.subheader(f"Top {k} probabilities")
st.table({
    "age_bin": [pretty_age_bin(classes[i]) for i in top_idx],
    "prob": [float(probs[i]) for i in top_idx],
})

# ---- Grad-CAM ----
if show_cam:
    st.subheader("Grad-CAM heatmap (explainability)")
    heatmap = cam(x, class_idx=pred_idx)
    gray = tensor_to_display_gray(x)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,5))
    plt.imshow(gray, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.title(f"Grad-CAM: {pred_label} ({pred_conf:.2f})")
    st.pyplot(fig)

st.markdown("---")
st.caption(
    "Disclaimer: This is a research/educational prototype. "
    "Predictions may be wrong. Do not use for medical decisions."
)
