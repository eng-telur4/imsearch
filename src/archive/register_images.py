import os
import glob
import psycopg2
from im2vec.vec2str import vec2str
from im2vec.paths import stored_img_path
from im2vec.natural_keys import natural_keys
from im2vec.image2vec import image2vec
from im2vec.normalize_vector import normalize_vector


if __name__ == "__main__":
    dsn = "dbname=postgres host=localhost user=postgres password=y6KDgfg9"
    image_files = sorted(glob.glob(f"{stored_img_path}/*"), key=natural_keys)
    vecs = image2vec(image_files, batch_size=32)  # 画像からベクトルに変換
    file_names = [os.path.splitext(os.path.basename(fp))[0] for fp in image_files]
    file_exts = [
        os.path.splitext(os.path.basename(fp))[1].split(".")[1] for fp in image_files
    ]
    vecs = vecs.astype("float32")  # float32に変換(FAISSに合わせるため)
    vecs = normalize_vector(vecs)  # 正規化
    print(vecs.shape)  # (500, 1280)
    vecs_s = vec2str(vecs)  # ベクトルを文字列に変換
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
                sql = "INSERT INTO images2 (image_data, file_name, file_type, embedding) VALUES (%s, %s, %s, %s);"
                cur.execute(
                    sql,
                    (
                        psycopg2.Binary(binary_data),
                        file_names[idx],
                        file_exts[idx],
                        vec_s,
                    ),
                )  # 500個のベクトルをDBに挿入
