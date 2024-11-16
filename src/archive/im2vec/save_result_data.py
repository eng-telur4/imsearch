import os
from archive.im2vec.paths import result_data_path


def save_result_data(file_names, queries, query_files, distances, indices, k=10):
    query_file = None
    result_file = None
    neighbor_idx = None
    similarity = None
    neighbor_file = None
    # 結果を保存するディレクトリを作成
    os.makedirs(result_data_path, exist_ok=True)

    for i in range(len(queries)):
        query_file = os.path.basename(query_files[i])

        # 結果をテキストファイルに保存
        result_file = f"{result_data_path}/{os.path.splitext(query_file)[0]}_result.csv"
        with open(result_file, "w") as f:
            f.write("Rank,FileName,Similarity\n")  # ヘッダー追加
            for j in range(k):
                neighbor_idx = indices[i][j]
                similarity = distances[i][j]  # 内積がコサイン類似度に対応
                neighbor_file = file_names[neighbor_idx]
                f.write(f"{j + 1},{neighbor_file},{similarity}\n")
        print(f"隣接データを '{result_file}' に保存しました。")
