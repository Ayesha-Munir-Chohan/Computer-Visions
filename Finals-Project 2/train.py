import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tensorflow as tf

from src.data import build_datasets
from src.model import baseline_autoencoder, transfer_autoencoder, combined_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=["tl", "baseline"],
        default="tl",
        help="tl = MobileNetV2 transfer autoencoder; baseline = skip-linked CNN (retrain after architecture changes).",
    )
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Frozen-encoder epochs before fine-tuning.")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--max_pairs", type=int, default=0, help="Use only first N paired images (0 = all).")
    parser.add_argument("--damage_augment", action="store_true", help="Train with synthetic damage augmentation.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.arch == "baseline":
        model_path = Path(args.out_dir) / "restorer_autoencoder.keras"
    else:
        model_path = Path(args.out_dir) / "restorer_autoencoder_tl.keras"

    train_ds, val_ds = build_datasets(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=0.1,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        damage_augment=args.damage_augment,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), save_best_only=True, monitor="val_loss", verbose=1
        ),
    ]

    if args.arch == "baseline":
        model = baseline_autoencoder(input_shape=(args.image_size, args.image_size, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=combined_loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
        print("Architecture: baseline CNN autoencoder")
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    else:
        model = transfer_autoencoder(
            input_shape=(args.image_size, args.image_size, 1),
            freeze_encoder=True,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=combined_loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        warmup_epochs = max(0, min(args.warmup_epochs, args.epochs))
        if warmup_epochs > 0:
            print(f"Stage 1: frozen encoder for {warmup_epochs} epoch(s)")
            model.fit(train_ds, validation_data=val_ds, epochs=warmup_epochs, callbacks=callbacks)

        if args.epochs > warmup_epochs:
            print(f"Stage 2: fine-tuning encoder for {args.epochs - warmup_epochs} epoch(s)")
            model.trainable = True
            model.compile(
                optimizer=tf.keras.optimizers.Adam(2e-4),
                loss=combined_loss,
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
            )
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                initial_epoch=warmup_epochs,
                callbacks=callbacks,
            )
        print("Architecture: transfer-learning autoencoder")

    if model_path.is_file():
        bak_dir = Path(args.out_dir) / "backups"
        bak_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = bak_dir / f"{model_path.stem}_{stamp}.keras"
        shutil.copy2(model_path, bak)
        print(f"Safety backup (restore from here if you delete the main file): {bak}")

    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
