import os
import torch
from PIL import Image
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

_ = load_dotenv(find_dotenv())

login(os.getenv("HF_TOKEN"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "stabilityai/japanese-stable-clip-vit-l-16"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)


def compute_image_embeddings(image):
    """
    画像の埋め込みベクトルを計算する関数。

    Args:
        image (PIL.Image.Image): 埋め込みを計算する画像。

    Returns:
        torch.Tensor: 正規化された画像の埋め込みベクトル。

    処理の流れ:
    1. 画像をモデルの入力形式に変換し、デバイスに移動。
    2. 勾配計算を無効化して、モデルを使用して画像特徴量を抽出。
    3. 特徴量ベクトルを正規化。
    4. 不要なメモリを解放。
    5. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    del image
    return image_features.cpu().detach()


if __name__ == "__main__":
    image = Image.open("/usr/src/app/image_data/stored_images/Car001.jpg")
    vec = compute_image_embeddings(image)
    print(type(vec))
    print(vec[0].shape)
    vec_np = vec.numpy()
    print(type(vec_np))
    print(vec_np[0].shape)
