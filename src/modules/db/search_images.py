import psycopg2
import numpy as np
from modules.gradio.vec2str import vec2str
from modules.gradio.compute_text_embeddings import compute_text_embeddings
from modules.gradio.compute_image_embeddings import compute_image_embeddings
from modules.settings import IMAGES_PER_PAGE, DSN, search_sql, search_sql_clip


# ? search_wrapperのヘルパー関数
# todo DB操作あり
# todo global変数あり
# @spaces.GPU(duration=60)
def search_images(query, search_method, limit=IMAGES_PER_PAGE):
    global IMAGES_PER_PAGE
    global DSN
    global search_sql
    global search_sql_clip

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:

            if search_method == "自然言語ベクトル検索":
                query_vec_str = []
                query_vec_str.append(compute_text_embeddings(query).numpy()[0])
                query_vec_str = np.array(query_vec_str).astype("float32")
                query_vec_str = query_vec_str / np.linalg.norm(
                    query_vec_str, axis=1, keepdims=True
                )
                query_vec_str = vec2str(query_vec_str)[0]
                cur.execute(
                    search_sql_clip,
                    (query_vec_str, limit),
                )
            elif search_method == "画像ベクトル検索":
                query_vec_str = compute_image_embeddings(query)
                cur.execute(
                    search_sql,
                    (query_vec_str, limit),
                )
            else:
                raise ValueError("無効な検索方法です")

            results = cur.fetchall()  # tuple[image_id, file_name, score, method]

            # LOBオブジェクトを文字列に変換
            processed_results = []
            for row in results:
                (
                    image_id,
                    file_name,
                    score,
                    method,
                ) = row
                processed_results.append(
                    (
                        image_id,
                        file_name,
                        score,
                        method,
                    )
                )

    print(f"検索結果: {len(processed_results)}件")
    return processed_results
