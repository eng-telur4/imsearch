import os
import re
import ftfy
import html
import torch
from PIL import Image
from typing import Union, List
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature

_ = load_dotenv(find_dotenv())

login(os.getenv("HF_TOKEN"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stabilityai/japanese-stable-clip-vit-l-16"
MODEL = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_PATH)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def tokenize(
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    global TOKENIZER

    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = TOKENIZER(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[TOKENIZER.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )


def compute_text_embeddings(text):
    """
    テキストの埋め込みベクトルを計算する関数。

    Args:
        text (str or List[str]): 埋め込みを計算するテキスト。単一の文字列または文字列のリスト。

    Returns:
        torch.Tensor: 正規化されたテキストの埋め込みベクトル。

    処理の流れ:
    1. 入力が単一の文字列の場合、リストに変換。
    2. テキストをトークン化。
    3. モデルを使用してテキスト特徴量を抽出。
    4. 特徴量ベクトルを正規化。
    5. 不要なメモリを解放。
    6. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    global MODEL
    global DEVICE

    if isinstance(text, str):
        text = [text]
    text = tokenize(texts=text)
    text_features = MODEL.get_text_features(**text.to(DEVICE))
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text
    return text_features.cpu().detach()


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
    global MODEL
    global DEVICE
    global PROCESSOR

    image = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = MODEL.get_image_features(**image)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    del image
    return image_features.cpu().detach()


if __name__ == "__main__":
    print("========== 画像ベクトルの生成 ==========")
    image = Image.open("/usr/src/app/image_data/stored_images/Car001.jpg")
    vec = compute_image_embeddings(image)
    print(type(vec))
    print(vec[0].shape)
    vec_np = vec.numpy()
    print(type(vec_np))
    print(vec_np[0].shape)

    print("========== 自然言語ベクトルの生成 ==========")
    text = "白い軽自動車が公道を走る"
    vec = compute_text_embeddings(text)
    print(type(vec))
    print(vec[0].shape)
    vec_np = vec.numpy()
    print(type(vec_np))
    print(vec_np[0].shape)
