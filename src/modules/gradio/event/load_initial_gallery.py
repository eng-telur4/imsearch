import gradio as gr
from modules.gradio.app.get_total_pages import get_total_pages
from modules.gradio.load_initial_images import load_initial_images


def load_initial_gallery():
    images, image_info = load_initial_images(1)
    total_pages = get_total_pages()  # 32
    page_info_text = f"1 / {total_pages}"
    # * ギャラリーを初期状態にする
    # * イメージの情報を初期画像の情報にする
    # * 「前」ボタンを無効にする
    # * 全ページ数が1ページより大きい場合は「次」ボタンを有効にする
    # * 現在のページ番号を1にする
    # * ページ情報を初期状態にする
    return (
        images,  # gallery
        image_info,  # image_info_state
        gr.update(interactive=False),  # prev_button
        gr.update(interactive=total_pages > 1),  # next_button
        1,  # current_page
        page_info_text,  # page_info
        gr.update(interactive=False),  # text_input
        gr.update(interactive=True),  # image_input
    )
