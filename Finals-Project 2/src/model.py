import tensorflow as tf
from tensorflow.keras import layers, models


def baseline_autoencoder(input_shape=(128, 128, 1)):
    """
    Gray → RGB with encoder–decoder skip links (detail-preserving) and
    upsample+conv (less checkerboard than strided transposed convs). Trains on CPU;
    retrain after changing this definition — old .keras weights will not load.
    """
    inp = layers.Input(shape=input_shape, name="gray_input")

    def enc_down(x, channels):
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(channels, 3, strides=2, padding="same", activation="relu")(x)
        return x

    x = enc_down(inp, 48)
    sk0 = x
    x = enc_down(x, 96)
    sk1 = x
    x = enc_down(x, 192)
    sk2 = x
    x = enc_down(x, 256)
    sk3 = x
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(sk3)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    def dec_up(x, skip, channels):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        return x

    x = dec_up(x, sk2, 192)
    x = dec_up(x, sk1, 128)
    x = dec_up(x, sk0, 96)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(3, 1, activation="sigmoid", name="rgb_output")(x)
    return models.Model(inp, out, name="baseline_autoencoder_restorer")


def transfer_autoencoder(input_shape=(128, 128, 1), freeze_encoder=True):
    """
    Transfer-learning autoencoder using MobileNetV2 encoder features.
    """
    inp = layers.Input(shape=input_shape, name="gray_input")
    x3 = layers.Concatenate(name="gray_to_rgb")([inp, inp, inp])

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = not freeze_encoder

    # Multi-scale features for detail-preserving decoder.
    feat_names = [
        "block_1_expand_relu",   # 64x64
        "block_3_expand_relu",   # 32x32
        "block_6_expand_relu",   # 16x16
        "block_13_expand_relu",  # 8x8
        "out_relu",              # 4x4
    ]
    encoder = models.Model(
        backbone.input,
        [backbone.get_layer(n).output for n in feat_names],
        name="mobilenetv2_encoder_features",
    )
    s0, s1, s2, s3, b = encoder(x3, training=False)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(b)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s3])
    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s2])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s1])
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, s0])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    out = layers.Conv2D(3, 1, activation="sigmoid", name="rgb_output")(x)
    return models.Model(inp, out, name="tl_autoencoder_restorer")


def combined_loss(y_true, y_pred):
    """L1 + SSIM; slightly higher SSIM weight helps crisp edges for restoration."""
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.75 * l1 + 0.25 * ssim
