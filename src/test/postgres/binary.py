import psycopg2

dsn = "dbname=postgres host=localhost user=postgres password=y6KDgfg9"

with psycopg2.connect(dsn) as conn:  # DBに接続する
    with conn.cursor() as cur:  # カーソルを開く
        # SQL文でbyteaデータを取得
        select_query = (
            "SELECT image_data, file_name, file_type FROM images WHERE image_id = 1;"
        )
        cur.execute(select_query)

        # データの取得
        record = cur.fetchone()
        if record is not None:
            image_data, file_name, fila_type = record  # byteaカラムのデータが取得される

            # バイナリデータをファイルに保存（例: 画像ファイル）
            with open(
                f"/usr/src/app/test/postgres/{file_name}.{fila_type}", "wb"
            ) as file:
                file.write(image_data)
