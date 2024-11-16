import psycopg2
from modules.settings import IMAGES_PER_PAGE, DSN, load_sql


# ? load_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# * 最新の画像をoffsetからlimit分取得する
# * タプル(image_id, file_name)をリストで返す
def get_latest_images(limit=IMAGES_PER_PAGE, offset=0) -> list[tuple[int, str]]:
    global IMAGES_PER_PAGE
    global DSN
    global load_sql

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:

            cur.execute(
                load_sql,
                (limit, offset),
            )

            results = cur.fetchall()

            # LOBオブジェクトを文字列に変換
            processed_results = []
            for row in results:
                image_id, file_name = row
                processed_results.append(
                    (
                        image_id,
                        file_name,
                    )
                )

    return processed_results
