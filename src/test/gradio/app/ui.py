import gradio as gr


with gr.Blocks(title="画像検索") as demo:
    image_info_state = None
    current_page = None
    search_method = None
    text_input = None
    search_button = None
    clear_button = None
    image_input = None
    gallery = None
    prev_button = None
    page_info = None
    next_button = None
    file_name = None
    distance = None

    image_info_state = gr.State([])
    current_page = gr.State(1)

    gr.Markdown("# マルチモーダル画像検索")
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                with gr.Column():
                    search_method = gr.Radio(
                        ["画像ベクトル検索", "自然言語ベクトル検索"],
                        label="検索方法",
                        value="画像ベクトル検索",
                    )
            with gr.Row():
                text_input = gr.Textbox(
                    label="検索テキスト",
                    lines=2,
                    show_copy_button=True,
                    interactive=True,
                )
            with gr.Row():
                with gr.Column(scale=6):
                    search_button = gr.Button("検索", variant="primary")
                with gr.Column(scale=1):
                    clear_button = gr.Button("クリア")
        with gr.Column(scale=2):
            image_input = gr.Image(
                label="検索画像",
                type="pil",
                height=280,
                width=500,
                interactive=False,
            )
    with gr.Row():
        with gr.Column(scale=7):
            gallery = gr.Gallery(
                label="検索結果",
                show_label=False,
                elem_id="gallery",
                columns=[8],
                rows=[2],
                height=380,
                interactive=False,
                show_download_button=True,
            )
    with gr.Row():
        prev_button = gr.Button("前")
        page_info = gr.Textbox(
            show_label=False,
            container=False,
            interactive=False,
        )
        next_button = gr.Button("次")
    with gr.Row():
        with gr.Column(scale=1):
            file_name = gr.Textbox(
                label="ファイル名",
                show_copy_button=True,
                interactive=False,
            )
            distance = gr.Textbox(
                label="ベクトル距離",
                show_copy_button=True,
                interactive=False,
            )


if __name__ == "__main__":
    try:
        demo.queue()
        demo.launch(debug=True, share=True, server_port=8899)
    except Exception as e:
        print(e)
        demo.close()
