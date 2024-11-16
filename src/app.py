import gradio as gr
from modules.gradio.event.on_select import on_select
from modules.gradio.event.prev_page import prev_page
from modules.gradio.event.next_page import next_page
from modules.gradio.event.search_wrapper import search_wrapper
from modules.gradio.event.clear_components import clear_components
from modules.gradio.event.update_text_input import update_text_input
from modules.gradio.event.update_image_input import update_image_input
from modules.gradio.event.load_initial_gallery import load_initial_gallery
from modules.settings import (
    SEARCH_IMAGE_HEIGHT,
    GALLERY_COLUMNS,
    GALLERY_ROWS,
    GALLERY_HEIGHT,
    DEVICE,
)


with gr.Blocks(title="画像検索") as demo:
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
                height=SEARCH_IMAGE_HEIGHT,
                interactive=False,
            )
    with gr.Row():
        with gr.Column(scale=7):
            gallery = gr.Gallery(
                label="検索結果",
                show_label=False,
                elem_id="gallery",
                columns=[GALLERY_COLUMNS],
                rows=[GALLERY_ROWS],
                height=GALLERY_HEIGHT,
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

    # ? イベントリスナ：Gradioアプリ(demo)起動時
    demo.load(
        load_initial_gallery,
        outputs=[
            gallery,
            image_info_state,
            prev_button,
            next_button,
            current_page,
            page_info,
            text_input,
            image_input,
        ],
    )

    # ? イベントリスナ：「検索方法」の項目変更時
    search_method.change(
        update_text_input,
        inputs=[
            search_method,
        ],
        outputs=[
            text_input,
        ],
    )
    search_method.change(
        update_image_input,
        inputs=[
            search_method,
        ],
        outputs=[
            image_input,
        ],
    )

    # ? イベントリスナ：「検索」ボタン押下時
    search_button.click(
        search_wrapper,
        inputs=[
            text_input,
            image_input,
            search_method,
        ],
        outputs=[
            gallery,
            image_info_state,
            text_input,
            image_input,
            search_button,
            gallery,
            prev_button,
            next_button,
            page_info,
        ],
    )

    # ? イベントリスナ：「クリア」ボタン押下時
    clear_button.click(
        clear_components,
        inputs=[
            search_method,
        ],
        outputs=[
            text_input,
            image_input,
            search_button,
            gallery,
            image_info_state,
            file_name,
            distance,
        ],
    )

    # ? イベントリスナ：「ギャラリー」の項目選択時
    gallery.select(
        on_select,
        inputs=[
            image_info_state,
        ],
        outputs=[
            file_name,
            distance,
        ],
    )

    # ? イベントリスナ：「前」ボタン押下時
    prev_button.click(
        prev_page,
        inputs=[
            current_page,
        ],
        outputs=[
            gallery,
            image_info_state,
            prev_button,
            next_button,
            current_page,
            page_info,
        ],
    )

    # ? イベントリスナ：「次」ボタン押下時
    next_button.click(
        next_page,
        inputs=[
            current_page,
        ],
        outputs=[
            gallery,
            image_info_state,
            prev_button,
            next_button,
            current_page,
            page_info,
        ],
    )

if __name__ == "__main__":
    print(f"device: {DEVICE}")
    try:
        demo.queue()
        demo.launch(debug=True, share=True)
    except Exception as e:
        print(e)
        demo.close()
