import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from restore import read_color
from src.utils import (
    postprocess_with_opencv,
    make_comparison_strip,
    clean_gray_input,
    remove_scratches_from_rgb,
    boost_color,
)

try:
    from streamlit_image_comparison import image_comparison
except Exception:
    image_comparison = None


st.set_page_config(page_title="AI-Based Photo Restoration", layout="wide")
st.title("AI-Based Photo Restoration")
st.caption("TensorFlow + OpenCV | CPU-friendly | Transfer-Learning Autoencoder restoration")


ROOT = Path(".")
DATASET_DIR = ROOT / "dataset"
GRAY_DIR = DATASET_DIR / "gray"
COLOR_DIR = DATASET_DIR / "color"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_BASELINE = ARTIFACTS_DIR / "restorer_autoencoder.keras"
MODEL_TL = ARTIFACTS_DIR / "restorer_autoencoder_tl.keras"
OUTPUTS_DIR = ROOT / "outputs"


def checkpoint_options():
    """Baseline listed first — often subjectively sharper after the same postprocess."""
    opts = []
    if MODEL_BASELINE.exists():
        opts.append(("Baseline CNN (skip links)", str(MODEL_BASELINE)))
    if MODEL_TL.exists():
        opts.append(("Transfer-learning MobileNetV2", str(MODEL_TL)))
    return opts


def model_file_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def list_jpgs(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.jpg"))


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, file_mtime: float):
    _ = file_mtime  # bust cache when the file on disk is replaced or retrained
    return tf.keras.models.load_model(model_path, compile=False)


def restore_single(gray_img_u8: np.ndarray, model, image_size: int):
    gray_resized = cv2.resize(gray_img_u8, (image_size, image_size), interpolation=cv2.INTER_AREA)
    gray_f = gray_resized.astype(np.float32) / 255.0
    gray_f = clean_gray_input(gray_f, sensitivity=0.48)
    inp = gray_f[None, ..., None]
    pred = model.predict(inp, verbose=0)[0]
    enhanced = postprocess_with_opencv(gray_f, pred)
    enhanced = remove_scratches_from_rgb(gray_f, enhanced, sensitivity=0.48)
    enhanced = boost_color(enhanced, sat_gain=1.28, val_gain=1.04)
    panel = make_comparison_strip(gray_f, pred, enhanced)
    return gray_f, pred, enhanced, panel


def get_model_image_size(model):
    return int(model.input_shape[1])


def zip_output_files(files):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    mem.seek(0)
    return mem


_ckpt = checkpoint_options()
if not _ckpt:
    st.error(
        "No model weights found in `artifacts/`. Train one with:\n"
        "`py -3.13 train.py --arch baseline ...` or `--arch tl` (default)."
    )
    st.stop()

st.sidebar.subheader("Model checkpoint")
_ckpt_labels = [x[0] for x in _ckpt]
_ckpt_paths = {x[0]: x[1] for x in _ckpt}
_picked = st.sidebar.radio("Weights", _ckpt_labels, index=0)
model_path = _ckpt_paths[_picked]

tab_single, tab_batch = st.tabs(["Single Image Restore", "Batch Restore"])


with tab_single:
    st.subheader("Single Image Restoration")
    image_size = 128
    keep_original_resolution = False
    st.caption(f"Using: **{_picked}** — OpenCV enhancement after the network.")

    uploaded = st.file_uploader("Upload grayscale image (.jpg/.png)", type=["jpg", "jpeg", "png"])
    use_dataset_sample = st.checkbox("Or use sample from dataset/gray", value=True)

    sample_img = None
    if use_dataset_sample:
        gray_files = list_jpgs(GRAY_DIR)
        if gray_files:
            chosen = st.selectbox("Choose sample image", [p.name for p in gray_files[:300]], index=0)
            sample_img = str(GRAY_DIR / chosen)
        else:
            st.warning("No sample image found in dataset/gray.")

    if st.button("Run Single Restore"):
        if not Path(model_path).exists():
            st.error("Model file not found. Train autoencoder first.")
        else:
            if uploaded is None and not sample_img:
                st.error("Please upload an image or select a dataset sample.")
            else:
                model = load_model_cached(model_path, model_file_mtime(model_path))
                image_size = get_model_image_size(model)
                if uploaded is not None:
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    gray_u8 = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                else:
                    gray_u8 = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)

                if gray_u8 is None:
                    st.error("Could not read selected image.")
                else:
                    gray_f, pred, enhanced, _ = restore_single(gray_u8, model, image_size)

                    if keep_original_resolution:
                        h0, w0 = gray_u8.shape[:2]
                        gray_show = cv2.resize(gray_f, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        pred_show = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        enhanced_show = cv2.resize(enhanced, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    else:
                        gray_show, pred_show, enhanced_show = gray_f, pred, enhanced

                    panel = make_comparison_strip(gray_show, pred_show, enhanced_show)
                    c1, c2, c3 = st.columns(3)
                    c1.image(np.repeat(gray_show[..., None], 3, axis=2), caption="Input Gray", use_container_width=True)
                    c2.image(pred_show, caption="Model Output", use_container_width=True)
                    c3.image(enhanced_show, caption="Final Enhanced", use_container_width=True)
                    st.image(panel, caption="Comparison Strip", use_container_width=True)
                    st.markdown("### Draggable Before/After")
                    before_img = np.repeat(gray_show[..., None], 3, axis=2)
                    after_img = np.clip(enhanced_show, 0, 1)
                    before_img_u8 = (np.clip(before_img, 0, 1) * 255).astype(np.uint8)
                    after_img_u8 = (np.clip(after_img, 0, 1) * 255).astype(np.uint8)
                    if image_comparison is not None:
                        image_comparison(
                            img1=before_img_u8,
                            img2=after_img_u8,
                            label1="Before (Gray/Damaged)",
                            label2="After (Restored)",
                            width=720,
                            show_labels=True,
                        )
                    else:
                        st.info(
                            "Install `streamlit-image-comparison` for draggable comparison. "
                            "Showing side-by-side fallback."
                        )
                        c4, c5 = st.columns(2)
                        c4.image(before_img, caption="Before", use_container_width=True)
                        c5.image(after_img, caption="After", use_container_width=True)

                    out_name = "single_restore_compare.jpg"
                    out_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                    ok, enc = cv2.imencode(".jpg", out_bgr)
                    if ok:
                        st.download_button(
                            "Download Comparison Image",
                            data=enc.tobytes(),
                            file_name=out_name,
                            mime="image/jpeg",
                        )


with tab_batch:
    st.subheader("Batch Restore + Metrics")
    image_size = 128
    save_original_res = True
    output_dir = str(OUTPUTS_DIR)
    save_scale = 0.85

    gray_files_all = list_jpgs(GRAY_DIR)
    gray_files = []

    if not gray_files_all:
        st.warning("No grayscale images found in dataset/gray.")
    else:
        pick_mode = st.radio(
            "How to choose images",
            ["Manual selection", "First N from dataset"],
            horizontal=True,
        )

        if pick_mode == "Manual selection":
            all_names = [p.name for p in gray_files_all]
            selected_names = st.multiselect(
                "Select images to restore",
                options=all_names,
                default=[],
                help="Type to search by filename, then pick one or more images.",
            )
            gray_files = [GRAY_DIR / n for n in selected_names]
            if selected_names:
                st.caption(f"**{len(selected_names)}** image(s) selected.")
            else:
                st.info("Pick at least one image above before running batch restore.")
        else:
            max_images = st.slider("Number of images", min_value=1, max_value=min(100, len(gray_files_all)), value=min(30, len(gray_files_all)))
            gray_files = gray_files_all[:max_images]
            st.caption(f"Will process the first **{len(gray_files)}** images from `dataset/gray`.")

    if st.button("Run Batch Restore"):
        if not Path(model_path).exists():
            st.error("Model file not found.")
        elif not gray_files:
            st.error("Please select at least one image for batch restore.")
        else:
            model = load_model_cached(model_path, model_file_mtime(model_path))
            image_size = get_model_image_size(model)
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            psnr_vals = []
            ssim_vals = []
            progress = st.progress(0, text="Batch restore started...")

            for i, p in enumerate(gray_files, start=1):
                src_u8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if src_u8 is None:
                    continue
                gray, pred, enhanced, _ = restore_single(src_u8, model, image_size)
                gt = read_color(str(COLOR_DIR / p.name), image_size)

                if save_original_res:
                    src_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if src_gray is not None:
                        h0, w0 = src_gray.shape[:2]
                        gray_show = cv2.resize(gray, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        pred_show = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        enhanced_show = cv2.resize(enhanced, (w0, h0), interpolation=cv2.INTER_LINEAR)
                        gt_show = None
                        gt_src = cv2.imread(str(COLOR_DIR / p.name), cv2.IMREAD_COLOR)
                        if gt_src is not None:
                            gt_src = cv2.cvtColor(gt_src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                            gt_show = cv2.resize(gt_src, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    else:
                        gray_show, pred_show, enhanced_show, gt_show = gray, pred, enhanced, gt
                else:
                    gray_show, pred_show, enhanced_show, gt_show = gray, pred, enhanced, gt

                strip = make_comparison_strip(gray_show, pred_show, enhanced_show, gt_show)
                if save_scale < 1.0:
                    h, w = strip.shape[:2]
                    strip = cv2.resize(strip, (int(w * save_scale), int(h * save_scale)), interpolation=cv2.INTER_AREA)

                save_file = out_path / f"{p.stem}_compare.jpg"
                cv2.imwrite(str(save_file), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

                if gt is not None:
                    gt_t = tf.convert_to_tensor(gt[None, ...], tf.float32)
                    en_t = tf.convert_to_tensor(enhanced[None, ...], tf.float32)
                    psnr_vals.append(float(tf.image.psnr(gt_t, en_t, max_val=1.0)[0].numpy()))
                    ssim_vals.append(float(tf.image.ssim(gt_t, en_t, max_val=1.0)[0].numpy()))

                pct = int((i / len(gray_files)) * 100)
                progress.progress(min(pct, 100), text=f"Processed {i}/{len(gray_files)} images")

            st.success(f"Saved batch outputs in: {out_path}")
            c1, c2 = st.columns(2)
            c1.metric("Mean PSNR", f"{np.mean(psnr_vals):.3f}" if psnr_vals else "N/A")
            c2.metric("Mean SSIM", f"{np.mean(ssim_vals):.3f}" if ssim_vals else "N/A")

            preview = sorted(out_path.glob("*_compare.jpg"))[:12]
            if preview:
                st.write("Preview")
                cols = st.columns(3)
                for idx, img_p in enumerate(preview):
                    img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
                    cols[idx % 3].image(img, caption=img_p.name, use_container_width=True)

            all_outputs = sorted(out_path.glob("*_compare.jpg"))
            if all_outputs:
                zip_bytes = zip_output_files(all_outputs)
                st.download_button(
                    "Download All Outputs (ZIP)",
                    data=zip_bytes,
                    file_name="restoration_outputs.zip",
                    mime="application/zip",
                )
