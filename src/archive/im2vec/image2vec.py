import keras
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# EfficientNetB0モデルをトップ層なしでロードし、平均プーリングを適用
model = keras.applications.EfficientNetB0(include_top=False, pooling="avg")


def image2vec(image_paths, batch_size=32):
    """
    画像パスのリストから特徴ベクトルを生成する関数（バッチ処理対応）
    """
    vecs = None
    num_images = None
    batch_paths = None
    images = None
    raw = None
    image = None
    vecs = []
    num_images = len(image_paths)
    for i in tqdm(range(0, num_images, batch_size), desc="ENETでの画像ベクトル化"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for path in batch_paths:
            raw = tf.io.read_file(path)
            image = tf.image.decode_jpeg(raw, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = keras.applications.efficientnet.preprocess_input(image)
            images.append(image)
        images = tf.stack(images)
        # バッチで予測、verbose=0でプログレスバーを非表示
        vec = model.predict(images, verbose=0)
        vecs.append(vec)
    vecs = np.vstack(vecs)
    return vecs
