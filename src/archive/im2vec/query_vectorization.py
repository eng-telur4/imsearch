import glob
import faiss
from archive.im2vec.paths import query_img_path
from im2vec.natural_keys import natural_keys
from im2vec.image2vec import image2vec


def query_vectorization(batch_size=32):
    queries = None
    query_files = None
    queries = []
    # クエリ画像を番号順に取得
    query_files = sorted(glob.glob(f"{query_img_path}/*"), key=natural_keys)
    print(f"クエリ画像の総数: {len(query_files)}")

    # バッチ処理でクエリベクトル抽出
    queries = image2vec(query_files, batch_size=batch_size)
    print("クエリ画像のベクトル化が完了しました。")

    queries = queries.astype("float32")  # FAISSはfloat32型を要求
    print(f"クエリベクトルの形状: {queries.shape}")

    # クエリベクトルを正規化
    faiss.normalize_L2(queries)

    return queries, query_files
