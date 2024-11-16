import glob
import psycopg2
from im2vec.vec2str import vec2str
from archive.im2vec.paths import query_img_path
from im2vec.natural_keys import natural_keys
from im2vec.image2vec import image2vec
from im2vec.normalize_vector import normalize_vector


if __name__ == "__main__":
    dsn = "dbname=postgres host=localhost user=postgres password=y6KDgfg9"
    query_files = sorted(glob.glob(f"{query_img_path}/*"), key=natural_keys)
    queries = image2vec(query_files, batch_size=32)  # 画像からベクトルに変換
    queries = queries.astype("float32")  # float32に変換(FAISSに合わせるため)
    queries = normalize_vector(queries)  # 正規化
    print(queries.shape)  # (18, 1280)
    queries_s = vec2str(queries)  # ベクトルを文字列に変換
    with psycopg2.connect(dsn) as conn:  # DBに接続する
        with conn.cursor() as cur:  # カーソルを開く
            sql = "select version()"
            cur.execute(sql)  # PostgreSQLのバージョン情報を取得
            print(cur.fetchone()[0])  # バージョン情報を表示
            for idx, query_s in enumerate(queries_s):
                print(f"({idx+1})")
                sql = f"SELECT *, 1 - (embedding <=> '{query_s}') AS cosine_similarity FROM imvecs ORDER BY cosine_similarity DESC LIMIT 10;"
                cur.execute(sql)
                # ret : list[tuple[id: int, embedding: str, cosine_similarity: float]]
                ret = cur.fetchall()
                print("--------+------------------------")
                print("id\t| cosine_similarity")
                print("--------+------------------------")
                for i in range(len(ret)):
                    print(f"{ret[i][0]}\t| {round(ret[i][2], 4)}")
                print("--------+------------------------")
