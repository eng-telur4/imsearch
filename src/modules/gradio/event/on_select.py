import gradio as gr


# ? イベントハンドラ：「ギャラリー」の項目選択時
# * input : (1) image_info_state
# * output : (1) file_name, distance
def on_select(evt: gr.SelectData, image_info):
    selected_index = None
    info = None
    vector_distance = None

    selected_index = evt.index
    if 0 <= selected_index < len(image_info):
        info = image_info[selected_index]
        vector_distance = info.get("vector_distance", "N/A")

        # * 選択された画像のファイル名を取得する
        # * 選択された画像のベクトル距離を取得する
        return (
            info["file_name"],  # file_name
            str(vector_distance),  # distance
        )
    else:
        # * 選択された画像のファイル名を選択エラーにする
        # * 選択された画像のベクトル距離をN/Aにする
        return (
            "選択エラー",  # file_name
            "N/A",  # distance
        )
