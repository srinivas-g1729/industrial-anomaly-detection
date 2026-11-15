"""
app.py
------
Streamlit application for PatchCore industrial anomaly detection.

Prerequisites:
    Run prepare_memory_banks.py first to build the ready memory banks:
        poetry run python scripts/prepare_memory_banks.py

Usage:
    poetry run streamlit run app/app.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.patchcore import (
    MultiScaleFeatureExtractor,
    extract_patch_embeddings,
    compute_patchwise_distances,
    patch_distances_to_maps,
)
from utils.normalization import preprocess_image

# Constants
READY_DIR = ROOT / "outputs" / "memory_banks" / "ready"
SAMPLE_DIR = ROOT / "app" / "sample_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

CATEGORY_DESCRIPTIONS = {
    "bottle": "Glass bottles — scratches, broken edges, contamination",
    "cable": "Electrical cables — cuts, missing strands, bent wires",
    "capsule": "Pharmaceutical capsules — cracks, discolouration, faulty printing",
    "carpet": "Carpet surface — colour defects, holes, thread errors",
    "grid": "Metal grids — broken wires, bent structures",
    "hazelnut": "Hazelnuts — cracks, holes, print defects",
    "leather": "Leather surface — colour spots, cuts, folding defects",
    "metal_nut": "Metal nuts — bent, flipped, scratched surfaces",
    "pill": "Pills — colour defects, cracks, contamination",
    "screw": "Screws — damaged threads, manipulated tips",
    "tile": "Ceramic tiles — cracks, glue spots, grey strokes",
    "toothbrush": "Toothbrushes — defective bristles",
    "transistor": "Transistors — damaged components, misplaced parts",
    "wood": "Wood panels — colour defects, holes, scratches",
    "zipper": "Zippers — broken teeth, fabric defects, split seams",
}

AUROC_REFERENCE = {
    "bottle": 0.9992,
    "cable": 0.9809,
    "capsule": 0.8855,
    "carpet": 0.9699,
    "grid": 0.8471,
    "hazelnut": 0.9989,
    "leather": 0.9912,
    "metal_nut": 0.9897,
    "pill": 0.8699,
    "screw": 0.8658,
    "tile": 0.9885,
    "toothbrush": 0.9889,
    "transistor": 0.9446,
    "wood": 0.9912,
    "zipper": 0.9443,
}

OPTIMAL_THRESHOLDS = {
    "bottle": 0.351,
    "cable": 0.402,
    "capsule": 0.296,
    "carpet": 0.347,
    "grid": 0.353,
    "hazelnut": 0.419,
    "leather": 0.338,
    "metal_nut": 0.385,
    "pill": 0.368,
    "screw": 0.349,
    "tile": 0.385,
    "toothbrush": 0.367,
    "transistor": 0.397,
    "wood": 0.383,
    "zipper": 0.327,
}

OVERLAY_ALPHA = 0.45
TOPK_RATIO = 0.01


# Cached resources


@st.cache_resource(show_spinner="Loading model...")
def load_extractor() -> MultiScaleFeatureExtractor:
    extractor = MultiScaleFeatureExtractor().to(DEVICE)
    extractor.eval()
    return extractor


@st.cache_resource(show_spinner="Loading reference data...")
def load_memory_bank(category: str) -> tuple:
    ready_path = READY_DIR / f"{category}_ready.pt"
    if not ready_path.exists():
        return None, {}
    saved = torch.load(ready_path, map_location="cpu")
    memory_bank = saved["memory_bank"]
    meta = {
        "full_size": saved.get("full_size", "?"),
        "coreset_ratio": saved.get("coreset_ratio", "?"),
        "coreset_size": memory_bank.shape[0],
        "embedding_dim": memory_bank.shape[1],
        "coreset_method": saved.get("coreset_method", "greedy"),
    }
    return memory_bank, meta


# Sample images (for visitors without a part image on hand)


def get_available_samples(category: str) -> dict:
    """Return which sample images exist for this category: {'normal': Path|None, 'defect': Path|None}."""
    samples = {}
    for kind in ("normal", "defect"):
        path = SAMPLE_DIR / f"{category}_{kind}.png"
        samples[kind] = path if path.exists() else None
    return samples


# Inference


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return preprocess_image(tensor).unsqueeze(0)


def run_inference(image_tensor, extractor, memory_bank):
    t0 = time.time()
    extractor.eval()
    with torch.no_grad():
        feature_map = extractor(image_tensor.to(DEVICE))
    B, C, H, W = feature_map.shape
    test_patches = extract_patch_embeddings(feature_map)
    test_patches = F.normalize(test_patches, dim=1)
    patch_distances = compute_patchwise_distances(
        test_patches=test_patches,
        memory_bank=memory_bank,
        chunk_size=1024,
    )
    anomaly_map = patch_distances_to_maps(patch_distances, B, H, W)
    flat = anomaly_map.view(1, -1)
    k = max(1, int(TOPK_RATIO * flat.shape[1]))
    raw_score = float(torch.topk(flat, k, dim=1).values.mean())
    score_norm = float(np.clip(raw_score / (raw_score + 1.0), 0, 1))
    anomaly_map_up = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    a_min, a_max = anomaly_map_up.min(), anomaly_map_up.max()
    heatmap = ((anomaly_map_up - a_min) / (a_max - a_min + 1e-8)).cpu().numpy()
    return score_norm, heatmap, time.time() - t0


def make_overlay(original: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    orig = (
        np.array(original.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    )
    colored = plt.colormaps["jet"](heatmap)[..., :3]
    blended = (1 - OVERLAY_ALPHA) * orig + OVERLAY_ALPHA * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# Result rendering


def render_result(original, score, heatmap, elapsed, threshold, category, filename):
    is_anomaly = score >= threshold

    st.markdown(f"#### `{filename}`")

    # Resize original to 256x256 for consistent display
    original_resized = original.convert("RGB").resize((256, 256))
    overlay_img = make_overlay(original, heatmap)

    # Legend bar above the 3 columns
    st.markdown(
        """
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px; font-size:0.82rem; color:#aaa;">
  <span>🌡️ Heatmap legend:</span>
  <span style="background:linear-gradient(to right,#00f,#0ff,#0f0,#ff0,#f00);
               width:120px; height:12px; border-radius:4px; display:inline-block;"></span>
  <span>🔵 Normal</span>
  <span>→</span>
  <span>🔴 Anomaly</span>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("📷 Original image")
        st.image(original_resized, use_container_width=True)
    with col2:
        st.caption("🌡️ Anomaly heatmap")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col3:
        st.caption("🔍 Defect overlay")
        st.image(overlay_img, use_container_width=True)

    st.markdown("---")
    score_col, verdict_col = st.columns([1, 2])

    with score_col:
        st.metric(
            label="Anomaly score",
            value=f"{score:.3f}",
            delta=f"threshold {threshold:.2f}",
            delta_color="off",
        )
        st.progress(float(score))
        st.caption(f"⏱️ Analysis time: {elapsed:.2f}s")

    with verdict_col:
        if is_anomaly:
            st.error(
                f"### ❌ Defective part detected\n"
                f"The anomaly score **{score:.3f}** exceeds the detection threshold **{threshold:.2f}**.\n\n"
                f"Check the heatmap — **red zones** indicate where the defect is located."
            )
        else:
            st.success(
                f"### ✅ No defect detected\n"
                f"The anomaly score **{score:.3f}** is below the threshold **{threshold:.2f}**.\n\n"
                f"The part appears **normal** compared to the reference database."
            )

    st.divider()


# Summary table (multi-image)


def render_summary(results: list):
    st.subheader("📊 Inspection summary")
    rows = []
    for filename, score, threshold, elapsed in results:
        verdict = "❌ Defective" if score >= threshold else "✅ Normal"
        rows.append(
            {
                "File": filename,
                "Anomaly score": round(score, 3),
                "Threshold": threshold,
                "Verdict": verdict,
                "Time (s)": round(elapsed, 2),
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    n_defective = sum(1 for r in results if r[1] >= r[2])
    n_normal = len(results) - n_defective
    c1, c2, c3 = st.columns(3)
    c1.metric("Images analysed", len(results))
    c2.metric("Defective", n_defective)
    c3.metric("Normal", n_normal)
    st.divider()


# Main


def main():
    st.set_page_config(
        page_title="Industrial Anomaly Detection",
        page_icon="🔍",
        layout="wide",
    )

    #  Welcome message
    st.title("🔍 Industrial Anomaly Detection")
    st.markdown(
        """
<div style="background-color:#1e3a5f; padding:16px; border-radius:8px; margin-bottom:16px; color:#eef2fa !important;">
<b style="color:#eef2fa;">What is this tool?</b><br>
<span style="color:#eef2fa;">This application automatically detects <b style="color:#eef2fa;">surface defects and anomalies</b> on industrial parts
using <b style="color:#eef2fa;">PatchCore</b>, a state-of-the-art deep learning method that compares each part
against a database of reference normal parts.</span><br><br>
<b style="color:#eef2fa;">Who is it for?</b> <span style="color:#eef2fa;">Quality control engineers, researchers, and students in computer vision
or industrial inspection.</span><br>
<b style="color:#eef2fa;">Dataset:</b> <span style="color:#eef2fa;">Trained and evaluated on <b style="color:#eef2fa;">MVTec AD</b>  the reference benchmark for industrial anomaly detection
(15 product categories, mean AUROC 0.95).</span>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 How to use — 3 steps", expanded=False):
        st.markdown("""
**Step 1 — Select the product category**
Choose the type of industrial part you want to inspect from the sidebar.
Each category has its own reference database of defect-free parts.

**Step 2 — Upload one or more images (or try a sample)**
Upload images of the parts to inspect (PNG or JPEG), or tick
"Try a sample image" if you don't have a part image on hand.

**Step 3 — Read the results**
For each image, the app displays:
- The **original image**
- A **heatmap** highlighting suspicious zones (red = anomaly, blue = normal)
- An **overlay** combining both views
- A **verdict** (Normal / Defective) with the anomaly score
""")

    st.divider()

    #  Sidebar
    with st.sidebar:
        st.header("🏭 Product category")

        category = st.selectbox(
            "Select the part type",
            options=ALL_CATEGORIES,
            index=2,
            format_func=lambda x: x.capitalize(),
        )

        st.caption(f"📦 {CATEGORY_DESCRIPTIONS[category]}")
        auroc = AUROC_REFERENCE[category]
        if auroc >= 0.95:
            perf = "🟢 Excellent"
        elif auroc >= 0.85:
            perf = "🟡 Good"
        else:
            perf = "🟠 Moderate"
        st.caption(
            f"{perf} Detection performance: **{auroc:.0%} of defects correctly detected** on benchmark data"
        )

        st.divider()
        st.markdown("**⚙️ Detection sensitivity**")
        threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=OPTIMAL_THRESHOLDS[category],
            step=0.01,
            help="Lower = more sensitive (catches more defects, more false alarms). "
            "Higher = more conservative (fewer false alarms, may miss subtle defects). "
            "Default is optimally calibrated for each category.",
        )
        st.caption(
            "🔽 Lower = more sensitive · 🔼 Higher = more conservative\n\n"
            "Default value is calibrated for this category."
        )

        st.divider()
        st.markdown("**💡 How it works**")
        st.caption(
            "The model extracts visual features from small patches of the image "
            "and compares each patch to a database of normal parts. "
            "If a patch looks very different from anything seen during training, "
            "it is flagged as anomalous — and highlighted in red on the heatmap."
        )

    #  Load models
    extractor = load_extractor()
    memory_bank, meta = load_memory_bank(category)

    if memory_bank is None:
        st.error(f"No reference database found for **{category}**.")
        st.info(
            "Run the preparation script first:\n"
            "```\npoetry run python scripts/prepare_memory_banks.py\n```"
        )
        st.stop()

    st.success(
        f"✅ Model ready for **{category.capitalize()}** "
        f"— reference database loaded ({meta['coreset_size']:,} reference patches)"
    )

    #  Image source: upload or sample
    files_to_process = []  # list of (file-like object, display filename)

    st.markdown("**No image on hand?**")
    samples = get_available_samples(category)
    has_samples = samples["normal"] or samples["defect"]

    use_sample = st.checkbox(
        "🖼️ Try a sample image from the MVTec AD dataset",
        disabled=not has_samples,
        help=(
            None if has_samples else "No sample image available for this category yet."
        ),
    )

    if use_sample and has_samples:
        available_kinds = [k for k, v in samples.items() if v is not None]
        sample_choice = st.radio(
            "Choose a sample:",
            options=available_kinds,
            format_func=lambda x: "Normal part" if x == "normal" else "Defective part",
            horizontal=True,
        )
        sample_path = samples[sample_choice]
        files_to_process.append((open(sample_path, "rb"), sample_path.name))
        st.caption(f"Using sample: `{sample_path.name}`")
    else:
        uploaded_files = st.file_uploader(
            "📂 Upload images to inspect",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload one or more images of the selected part type.",
        )
        if uploaded_files:
            files_to_process = [(f, f.name) for f in uploaded_files]

    if not files_to_process:
        st.info(
            "👆 Upload one or more images above, or try a sample, to start the inspection."
        )
        return

    st.divider()

    # Run inference
    all_results = []
    result_data = []

    for file_obj, filename in files_to_process:
        original = Image.open(file_obj)
        image_tensor = preprocess_pil(original)
        with st.spinner(f"Analysing `{filename}`..."):
            score, heatmap, elapsed = run_inference(
                image_tensor, extractor, memory_bank
            )
        all_results.append((filename, score, threshold, elapsed))
        result_data.append((filename, original, score, heatmap, elapsed))

    #  Summary (only if multiple images)
    if len(files_to_process) > 1:
        render_summary(all_results)

    #  Detailed results
    st.subheader(f"🔬 Detailed results — {len(files_to_process)} image(s)")
    for filename, original, score, heatmap, elapsed in result_data:
        render_result(
            original=original,
            score=score,
            heatmap=heatmap,
            elapsed=elapsed,
            threshold=threshold,
            category=category,
            filename=filename,
        )


if __name__ == "__main__":
    main()
