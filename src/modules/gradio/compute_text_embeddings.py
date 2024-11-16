from modules.gradio.tokenize import tokenize
from modules.settings import MODEL, DEVICE


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
