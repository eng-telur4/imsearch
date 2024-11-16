import os
import glob
import faiss
import pickle
from im2vec.natural_keys import natural_keys
from im2vec.image2vec import image2vec
from archive.im2vec.paths import db_file, file_names_file, stored_img_path


def stored_vectorization(batch_size=32):
    image_files = None
    vecs = None
    file_names = None
    dim = None
    index = None
    if not os.path.exists(db_file) or not os.path.exists(file_names_file):
        print(
            f"{db_file} 又は {file_names_file} が存在しません。ベクトル化を開始します。"
        )

        # 画像を番号順に取得
        image_files = sorted(glob.glob(f"{stored_img_path}/*"), key=natural_keys)
        print(f"画像ファイル数: {len(image_files)}")

        # バッチ処理で特徴量抽出
        vecs = image2vec(image_files, batch_size=batch_size)
        file_names = [os.path.basename(fp) for fp in image_files]
        print("ベクトル化が完了しました。")

        vecs = vecs.astype("float32")  # FAISSはfloat32型を要求
        print(f"ベクトルの形状: {vecs.shape}")

        # FAISS用にベクトルを正規化（コサイン類似度を内積として使用するため）
        faiss.normalize_L2(vecs)

        # FAISSのインデックス作成（内積用）
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # 内積を使用するインデックス
        index.add(vecs)  # ベクトルをインデックスに追加
        print(f"インデックスに追加されたベクトルの数: {index.ntotal}")

        # インデックスを保存
        faiss.write_index(index, db_file)
        print(f"FAISSインデックスを {db_file} に保存しました。")

        # ファイル名リストを保存
        with open(file_names_file, "wb") as f:
            pickle.dump(file_names, f)
        print(f"ファイル名リストを {file_names_file} に保存しました。")
    else:
        print(
            f"{db_file} および {file_names_file} が既に存在します。既存のファイルを使用します。"
        )
        # インデックスを読み込む
        index = faiss.read_index(db_file)
        print(f"FAISSインデックスを {db_file} から読み込みました。")

        # ファイル名リストを読み込む
        with open(file_names_file, "rb") as f:
            file_names = pickle.load(f)
        print(f"ファイル名リストを {file_names_file} から読み込みました。")

    return index, file_names
