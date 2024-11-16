import io
import gradio as gr
from PIL import Image
from modules.db.search_images import search_images
from modules.gradio.app.get_total_pages import get_total_pages
from modules.gradio.load_initial_images import load_initial_images
from modules.db.get_multiple_image_data import get_multiple_image_data


# ? イベントハンドラ：「検索」ボタン押下時
# * input : (3) text_input,image_input,search_method
# * output : (9) gallery,image_info_state,text_input,image_input,search_button,gallery,prev_button,next_button,page_info
def search_wrapper(text_query, image_query, search_method):
    # 検索テキストまたは検索画像があるなら
    if (text_query or image_query) != None:
        results = search_images(
            text_query if text_query else image_query,
            search_method,
        )

        # 検索結果が空の場合
        if not results:
            print("検索結果が空です")
            # * ギャラリーを空にする
            # * イメージの情報を空にする
            # * 検索テキストのTextboxを有効にする
            # * 検索画像を有効にする
            # * 検索ボタンを有効にする
            # * ギャラリーの画像選択状態をなしにする
            # * 「前」ボタンを無効にする
            # * 「次」ボタンを無効にする
            # * ページ情報を空にする
            return (
                [],  # gallery
                [],  # image_info_state
                gr.update(interactive=True),  # text_input
                gr.update(interactive=True),  # image_input
                gr.update(interactive=True),  # search_button
                gr.update(selected_index=None),  # gallery
                gr.update(interactive=False),  # prev_button
                gr.update(interactive=False),  # next_button
                "",  # page_info
            )
        # ベクトル検索結果と全文検索結果を分離
        vector_results = []
        vector_info = []

        # 画像IDのリストを作成
        image_ids = [result[0] for result in results]

        # 一度にすべての画像データを取得
        image_data_dict = get_multiple_image_data(image_ids)

        for result in results:
            (
                image_id,
                file_name,
                score,  # vector_distance | cosine_similarity
                method,
            ) = result

            image_data = image_data_dict.get(image_id)
            if image_data:
                img = Image.open(io.BytesIO(image_data))
                img.load()
            else:
                continue  # 画像データが見つからない場合はスキップ

            # 情報を構築
            info = {
                "image_id": image_id,
                "file_name": file_name,
                "vector_distance": score if score is not None else "N/A",
                "method": method,
            }

            caption = f"ベクトル距離: {round(float(score), 3) if isinstance(score, (int, float)) else score}"
            vector_results.append((img, caption))
            vector_info.append(info)

        # ギャラリーの画像と情報を統合
        gallery_images = vector_results
        image_info = vector_info

        print(f"search_method: {search_method}")

        page_info_text = ""

        # ギャラリーの更新と他のUI要素の更新
        # * ギャラリーを取得画像にする
        # * イメージの情報をギャラリーの情報にする
        # * 検索テキストのTextboxを空にし、自然言語ベクトル検索なら入力を有効にする
        # * 検索画像を空にし、画像ベクトル検索なら入力を有効にする
        # * 検索ボタンを有効にする
        # * ギャラリーの画像選択状態をなしにする
        # * 「前」ボタンを有効にする
        # * 「次」ボタンを有効にする
        # * ページ情報をギャラリーの状態にする
        return (
            gallery_images,  # gallery
            image_info,  # image_info_state
            gr.update(
                interactive=(search_method == "自然言語ベクトル検索")
            ),  # text_input
            gr.update(interactive=(search_method == "画像ベクトル検索")),  # image_input
            gr.update(interactive=True),  # search_button
            gr.update(selected_index=None),  # gallery
            gr.update(interactive=False),  # prev_button
            gr.update(interactive=False),  # next_button
            page_info_text,  # page_info
        )
    # 検索テキストまたは検索画像がないなら
    else:
        # 初期表示に戻す
        images, image_info = load_initial_images()
        gallery_images = [(img, None) for img in images]
        page_info_text = f"1 / {get_total_pages()}"
        # * ギャラリーを初期状態にする
        # * イメージの情報を初期画像の情報にする
        # * 検索テキストのTextboxの入力を有効にする
        # * 検索画像の入力を無効にする
        # * 検索ボタンを有効にする
        # * ギャラリーの画像選択状態をなしにする
        # * 「前」ボタンを有効にする
        # * 「次」ボタンを有効にする
        # * ページ情報を初期状態にする
        return (
            gallery_images,  # gallery
            image_info,  # image_info_state
            gr.update(interactive=True),  # text_input
            gr.update(interactive=False),  # image_input
            gr.update(interactive=True),  # search_button
            gr.update(selected_index=None),  # gallery
            gr.update(interactive=True),  # prev_button
            gr.update(interactive=True),  # next_button
            page_info_text,  # page_info
        )
