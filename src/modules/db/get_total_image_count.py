import psycopg2
from modules.settings import DSN


# ? get_total_pagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# * DBに登録されている画像数を取得する
def get_total_image_count() -> int:
    global DSN

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM images;")
            count = cur.fetchone()[0]
    return count
