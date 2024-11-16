import io
import keras
import numpy as np
import tensorflow as tf
from modules.settings import MODEL_ENET
from modules.gradio.vec2str import vec2str


# ? search_imagesのヘルパー関数
# todo global変数あり
def compute_image_embeddings(image):
    global MODEL_ENET

    # GradioのPIL画像をJPEG形式のバイト列に変換
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()

    # JPEGバイトデータをTensorFlowで読み込み
    img = tf.image.decode_jpeg(byte_data, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = keras.applications.efficientnet.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # バッチ次元を追加

    # ベクトル化と正規化
    vec = MODEL_ENET.predict(img)
    vec = vec.astype("float32")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec2str(vec)

    return vec[0]
