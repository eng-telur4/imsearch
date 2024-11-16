from time import time
from im2vec.save_result_data import save_result_data
from im2vec.save_result_images import save_result_images
from im2vec.query_vectorization import query_vectorization
from im2vec.stored_vectorization import stored_vectorization


if __name__ == "__main__":

    # バッチサイズの設定
    BATCH_SIZE = 32  # 必要に応じて調整

    # ? (1) 画像ベクトル化とインデックスの作成・保存
    start = time()
    index, file_names = stored_vectorization(batch_size=BATCH_SIZE)
    end = time()
    print(f"特徴量抽出にかかった時間：{end - start} 秒")

    # ? (2) クエリ画像のベクトル化
    queries, query_files = query_vectorization(batch_size=BATCH_SIZE)

    # ? (3) 上位10件の類似ベクトルを検索
    k = 10  # 取得する近傍の数
    distances, indices = index.search(queries, k)

    # ? (4) 結果を保存する
    save_result_data(file_names, queries, query_files, distances, indices, k=k)
    save_result_images()
