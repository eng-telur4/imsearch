import gradio as gr


# ? イベントハンドラ：「検索方法」の項目変更時
# * input : (1) search_method
# * output : (1) text_input
def update_image_input(search_method):
    # * 検索方法が画像ベクトル検索なら検索画像を有効にする
    if search_method == "画像ベクトル検索":
        return gr.update(interactive=True)  # image_input
    # * 検索方法が画像ベクトル検索なら検索画像を無効にする
    else:
        return gr.update(value=None, interactive=False)  # image_input
