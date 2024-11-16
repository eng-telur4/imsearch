import gradio as gr
from modules.gradio.app.get_total_pages import get_total_pages
from modules.gradio.load_initial_images import load_initial_images


# ? next_pageとprev_pageのヘルパー関数
def change_page(direction, current_page):
    total_pages = None
    images = None
    image_info = None
    gallery_images = None
    page_info_text = None

    total_pages = get_total_pages()

    if direction == "next" and current_page < total_pages:
        current_page += 1
    elif direction == "prev" and current_page > 1:
        current_page -= 1

    images, image_info = load_initial_images(current_page)
    gallery_images = [(img, None) for img in images]
    page_info_text = f"{current_page} / {total_pages}"
    # * ギャラリーを前または次のページの画像にする
    # * イメージの情報を前または次のページの画像の情報にする
    # * 現在のページ数が1より大きい場合は、「前」ボタンを有効にする
    # * 全ページ数が現在のページ番号より大きい場合は「次」ボタンを有効にする
    # * 現在のページ番号を現在のページ番号にする
    # * ページ情報を前または次のページの画像に合わせる
    return (
        gallery_images,
        image_info,
        gr.update(interactive=current_page > 1),
        gr.update(interactive=current_page < total_pages),
        current_page,
        page_info_text,
    )
