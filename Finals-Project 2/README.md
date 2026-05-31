# AI-Based Photo Restoration (CPU-Friendly)

This project restores grayscale human face photos into enhanced color photos using:
- TensorFlow (transfer-learning autoencoder model)
- OpenCV (detail-aware post-processing)

Dataset structure:
```
dataset/
  color/   # 8164 jpg files
  gray/    # 8164 jpg files
```

## Why this stands out
1. **Dual-stage restoration pipeline**:
   - Stage-1: CNN predicts colorized/restored output.
   - Stage-2: OpenCV edge-guided enhancement improves local details.
2. **Presentation-ready visual outputs**:
   - Generates side-by-side strips: input gray | model output | enhanced output | ground truth.
3. **Live demo mode**:
   - Artificially damages an image (noise + scratches) and restores it for a dramatic before/after presentation.

## Setup
```bash
pip install -r requirements.txt
```

## Run Streamlit Frontend
```bash
streamlit run app.py
```

The frontend includes:
- Single image upload, restoration preview, and draggable before/after comparison
- Batch restoration with PSNR/SSIM, gallery preview, ZIP download

## Train (CPU)

**Transfer-learning (default)** — saves `artifacts/restorer_autoencoder_tl.keras`:
```bash
python train.py --arch tl --dataset_dir dataset --image_size 128 --batch_size 8 --epochs 6 --warmup_epochs 2 --max_pairs 1200
```

**Baseline CNN** — smaller file, often looks sharper on faces; saves `artifacts/restorer_autoencoder.keras`:
```bash
python train.py --arch baseline --dataset_dir dataset --image_size 128 --batch_size 8 --epochs 8 --max_pairs 2000
```

Tip for slower laptops:
- Try `--batch_size 8`
- Start with `--epochs 4` for quick proof-of-concept
- Use subset training: `--max_pairs 1200` (or even `800`)

Python 3.13 command example (baseline checkpoint):
```bash
py -3.13 train.py --arch baseline --dataset_dir dataset --image_size 128 --batch_size 8 --epochs 8 --max_pairs 2000
```

## Restore and evaluate
```bash
python restore.py --model artifacts/restorer_autoencoder_tl.keras --input_dir dataset/gray --gt_dir dataset/color --max_images 30 --presentable_mode --save_scale 0.85
```

This prints mean PSNR/SSIM and saves visual strips in `outputs/`.

## Live presentation demo
```bash
python demo_degrade_and_restore.py --image dataset/gray/1001.jpg --out demo_result.jpg
```

Output panel:
- original gray
- damaged gray
- restored/enhanced

## Presentation flow
1. Problem statement: restoring old grayscale face photos.
2. Method: lightweight TensorFlow model + OpenCV enhancement.
3. Results: show PSNR/SSIM and side-by-side comparisons.
4. Unique value: explain edge-guided post-processing and damage-repair demo.
5. Future scope: super-resolution, GAN-based realism, mobile deployment.

## Workflow
1. Train baseline (or `--arch tl` for MobileNetV2): `py -3.13 train.py --arch baseline --dataset_dir dataset --image_size 128 --batch_size 8 --epochs 8 --max_pairs 2000`
2. Generate outputs: `py -3.13 restore.py --model artifacts/restorer_autoencoder.keras --input_dir dataset/gray --gt_dir dataset/color --max_images 30 --presentable_mode --save_scale 0.85` (or `--model artifacts/restorer_autoencoder_tl.keras`)
3. Start UI: `py -3.13 -m streamlit run app.py` — sidebar picks the checkpoint file.
