import argparse
import numpy as np
import cv2
import tensorflow as tf

from src.utils import postprocess_with_opencv


def add_old_photo_damage(gray: np.ndarray):
    h, w = gray.shape
    noisy = gray.copy()

    gauss = np.random.normal(0, 10, (h, w)).astype(np.float32)
    noisy = np.clip(noisy + gauss, 0, 255).astype(np.uint8)

    for _ in range(10):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(noisy, (x1, y1), (x2, y2), color=np.random.randint(200, 255), thickness=1)

    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)
    return noisy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/restorer_autoencoder_tl.keras")
    parser.add_argument("--image", required=True, help="Path to a grayscale face image")
    parser.add_argument("--out", default="demo_result.jpg")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (args.size, args.size))
    damaged = add_old_photo_damage(gray)

    inp = damaged.astype(np.float32) / 255.0
    pred = model.predict(inp[None, ..., None], verbose=0)[0]
    enhanced = postprocess_with_opencv(inp, pred)

    panel = np.concatenate(
        [
            np.repeat((gray / 255.0)[..., None], 3, axis=2),
            np.repeat((damaged / 255.0)[..., None], 3, axis=2),
            np.clip(enhanced, 0, 1),
        ],
        axis=1,
    )
    cv2.imwrite(args.out, cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Saved demo image: {args.out}")


if __name__ == "__main__":
    main()
