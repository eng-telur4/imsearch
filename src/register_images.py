import os
import glob
import pickle
import psycopg2
import numpy as np
from tqdm import tqdm
from PIL import Image
from modules.gradio.vec2str import vec2str
from modules.gradio.image2vec import image2vec
from modules.gradio.natural_keys import natural_keys
from modules.gradio.normalize_vector import normalize_vector
from modules.gradio.compute import compute_image_embeddings
from modules.paths import stored_img_path


def save_embeddings_to_pickle(vecs, filename):
    with open(filename, "wb") as f:
        pickle.dump(vecs, f)


def load_embeddings_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    dsn = "dbname=postgres host=localhost user=postgres password=y6KDgfg9"
    image_files = sorted(glob.glob(f"{stored_img_path}/*"), key=natural_keys)
    file_names = [os.path.splitext(os.path.basename(fp))[0] for fp in image_files]
    file_exts = [
        os.path.splitext(os.path.basename(fp))[1].split(".")[1] for fp in image_files
    ]

    # EffecientNetB0 ベクトルの処理
    vecs_filename = "efficientnetb0_embeddings.pkl"
    if os.path.exists(vecs_filename):
        print("EfficientNetB0の埋め込みベクトルを読み込み中...")
        vecs = load_embeddings_from_pickle(vecs_filename)
    else:
        print("EfficientNetB0の埋め込みベクトルを生成中...")
        vecs = image2vec(image_files, batch_size=32)  # 画像からベクトルに変換
        vecs = vecs.astype("float32")  # float32に変換(FAISSに合わせるため)
        vecs = normalize_vector(vecs)  # 正規化
        save_embeddings_to_pickle(vecs, vecs_filename)

    print(vecs.shape)  # (500, 1280)
    vecs_s = vec2str(vecs)  # ベクトルを文字列に変換

    # Japanese StableCLIPベクトルの処理
    vecs_clip_filename = "japanese_stable_clip_embeddings.pkl"
    if os.path.exists(vecs_clip_filename):
        print("CLIPの埋め込みベクトルを読み込み中...")
        vecs_clip = load_embeddings_from_pickle(vecs_clip_filename)
    else:
        print("CLIPの埋め込みベクトルを生成中...")
        vecs_clip = []
        for i in tqdm(range(len(image_files)), desc="CLIPでの画像ベクトル化"):
            image = Image.open(image_files[i])
            vec = compute_image_embeddings(image)
            vecs_clip.append(vec.numpy()[0])
        vecs_clip = np.array(vecs_clip).astype("float32")
        vecs_clip = normalize_vector(vecs_clip)
        save_embeddings_to_pickle(vecs_clip, vecs_clip_filename)

    print(vecs_clip.shape)
    vecs_clip_s = vec2str(vecs_clip)

    # データベースへの挿入
    with psycopg2.connect(dsn) as conn:  # DBに接続する
        with conn.cursor() as cur:  # カーソルを開く
            sql = "select version()"
            cur.execute(sql)  # PostgreSQLのバージョン情報を取得
            print(cur.fetchone()[0])  # バージョン情報を表示
            print("========== ========== ==========")
            for idx, vec_s in enumerate(vecs_s):
                file_path = f"{stored_img_path}/{file_names[idx]}.{file_exts[idx]}"
                print(file_path)
                with open(file_path, "rb") as file:
                    binary_data = file.read()
                sql = "INSERT INTO images (image_data, file_name, file_type, embedding, embedding_clip) VALUES (%s, %s, %s, %s, %s);"
                cur.execute(
                    sql,
                    (
                        psycopg2.Binary(binary_data),
                        file_names[idx],
                        file_exts[idx],
                        vec_s,
                        vecs_clip_s[idx],
                    ),
                )  # 500個のベクトルをDBに挿入
