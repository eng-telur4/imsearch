import psycopg2
from modules.settings import DSN


# ? search_wrapperとload_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# * 画像idリスト(image_ids)から画像id(image_id)と画像(image_data)を取得する
def get_multiple_image_data(image_ids) -> dict[int, bytes]:
    global DSN

    # 画像IDが空の場合は空の辞書を返す
    if not image_ids:
        return {}

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            placeholders = ",".join([f"%s" for _ in range(len(image_ids))])
            query = f"SELECT image_id, image_data FROM images WHERE image_id IN ({placeholders});"

            cur.execute(query, image_ids)
            rows = cur.fetchall()
            image_data_dict = {image_id: image_data for image_id, image_data in rows}

    return image_data_dict
