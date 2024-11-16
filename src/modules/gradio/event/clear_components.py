import gradio as gr


# ? イベントハンドラ：「クリア」ボタン押下時
# * input : (1) search_method
# * output : (7) text_input,image_input,search_button,gallery,image_info_state,file_name,distance
def clear_components(search_method):
    # * 検索テキストのTextboxを空にし、自然言語ベクトル検索なら入力を有効にする
    # * 検索画像を空にし、画像ベクトル検索なら入力を有効にする
    # * 検索ボタンを有効にする
    # * ギャラリーを空にする
    # * イメージの情報を空にする
    # * ファイル名のTextboxを空にする
    # * ベクトル距離のTextboxを空にする
    return (
        gr.update(
            value="", interactive=(search_method == "自然言語ベクトル検索")
        ),  # text_input
        gr.update(
            value=None, interactive=(search_method == "画像ベクトル検索")
        ),  # image_input
        gr.update(interactive=True),  # search_button
        gr.update(value=None, selected_index=None),  # gallery
        [],  # image_info_state
        gr.update(value=""),  # file_name
        gr.update(value=""),  # distance
    )
