import gradio as gr


# ? イベントハンドラ：「検索方法」の項目変更時
# * input : (1) search_method
# * output : (1) text_input
def update_text_input(search_method):
    # * 検索方法が画像ベクトル検索なら検索テキストのTextboxを無効にする
    if search_method == "画像ベクトル検索":
        return gr.update(value=None, interactive=False)  # text_input
    # * 検索方法が自然言語ベクトル検索なら検索テキストのTextboxを有効にする
    else:
        return gr.update(interactive=True)  # text_input
