import io
import re
import ftfy
import html
import keras
import torch
import psycopg2
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf
from typing import Union, List
from huggingface_hub import login
from PIL.ImageFile import ImageFile
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature

# ! ========== ========== ========== ========== ==========
# ! 定数
# ! ========== ========== ========== ========== ==========

# ! load_initial_images()
# ! get_total_pages()
# * ギャラリーに一度に表示する画像数
IMAGES_PER_PAGE = 16
GALLERY_ROWS = 2
GALLERY_COLUMNS = IMAGES_PER_PAGE // GALLERY_ROWS
if IMAGES_PER_PAGE % GALLERY_ROWS:
    GALLERY_COLUMNS += 1
GALLERY_HEIGHT = 380
GALLERY_IMAGE_HEIGHT = GALLERY_HEIGHT // GALLERY_ROWS
SEARCH_IMAGE_HEIGHT = 280

DBNAME = "postgres"
HOST = "localhost"
USER = "postgres"
PASSWORD = "y6KDgfg9"
DSN = f"dbname={DBNAME} host={HOST} user={USER} password={PASSWORD}"

# ! ========== ========== ========== ========== ==========
# ! グローバル関数
# ! ========== ========== ========== ========== ==========

# ? ========== ========== ========== ========== ==========
# ? グローバル関数
# ? ========== ========== ========== ========== ==========

t_vector_distance = Union[float, str]

t_images = list[ImageFile]
t_image_info = list[dict[str, t_vector_distance]]


# ? search_wrapperとchange_pageとload_initial_galleryのヘルパー関数
# todo global変数あり
def load_initial_images(
    page=1,
) -> tuple[t_images, t_image_info]:

    global IMAGES_PER_PAGE
    offset = None
    limit = None
    results = None
    image_ids = None
    image_data_dict = None
    images = None
    image_info = None
    image_id = None
    file_name = None
    image_data = None

    # * 最新の画像をoffsetからlimit分取得
    # offset(とばす画像数)とlimit(取得する画像数)を設定
    offset = (page - 1) * IMAGES_PER_PAGE  # (1-1) = 0 × 16 = 0
    limit = IMAGES_PER_PAGE  # 16
    results = get_latest_images(
        limit=limit, offset=offset
    )  # list[tuple[image_id: int, file_name: str]]

    # * 画像idリスト(image_ids)から画像(image_data)を取得する
    # 画像IDのリストを作成
    image_ids = [result[0] for result in results]  # list[image_id: int]
    print(image_ids)
    image_data_dict = get_multiple_image_data(
        image_ids
    )  # dict[image_id: int, image_data: byte]

    images = []
    image_info = []
    for image_id, file_name in results:
        # image_idからimage_dataを取得
        image_data = image_data_dict.get(image_id)
        images.append(Image.open(io.BytesIO(image_data)))
        image_info.append(
            {
                "file_name": file_name,
                "vector_distance": "N/A",
            }
        )

    return (images, image_info)


# ? search_wrapperとload_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# * 画像idリスト(image_ids)から画像id(image_id)と画像(image_data)を取得する
def get_multiple_image_data(image_ids) -> dict[int, bytes]:
    global DSN
    conn = None
    cur = None
    placeholders = None
    query = None
    image_data_dict = None

    # 画像IDが空の場合は空の辞書を返す
    if not image_ids:
        return {}

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            placeholders = ",".join([f"%s" for _ in range(len(image_ids))])
            query = f"SELECT image_id, image_data FROM images WHERE image_id IN ({placeholders});"

            cur.execute(query, image_ids)
            rows = cur.fetchall()
            image_data_dict = {image_id: image_data for image_id, image_data in rows}

    return image_data_dict


search_sql = """
SELECT image_id, file_name,
    1 - (embedding <=> %s) AS cosine_similarity,
    'vector' as method
FROM images ORDER BY cosine_similarity
DESC LIMIT %s;
"""
search_sql_clip = """
SELECT image_id, file_name,
    1 - (embedding_clip <=> %s) AS cosine_similarity,
    'vector' as method
FROM images ORDER BY cosine_similarity
DESC LIMIT %s;
"""


# ? search_wrapperのヘルパー関数
# todo DB操作あり
# todo global変数あり
# @spaces.GPU(duration=60)
def search_images(query, search_method, limit=IMAGES_PER_PAGE):
    global IMAGES_PER_PAGE
    global DSN
    global search_sql
    conn = None
    cur = None
    query_vec_str = None
    results = None
    processed_results = None
    image_id = None
    file_name = None
    score = None
    method = None
    row = None

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:

            if search_method == "自然言語ベクトル検索":
                query_vec_str = []
                query_vec_str.append(compute_text_embeddings(query).numpy()[0])
                query_vec_str = np.array(query_vec_str).astype("float32")
                query_vec_str = query_vec_str / np.linalg.norm(
                    query_vec_str, axis=1, keepdims=True
                )
                query_vec_str = vec2str(query_vec_str)[0]
                cur.execute(
                    search_sql_clip,
                    (query_vec_str, limit),
                )
                print("自然言語ベクトルで検索しました")
            elif search_method == "画像ベクトル検索":
                query_vec_str = compute_image_embeddings(query)
                cur.execute(
                    search_sql,
                    (query_vec_str, limit),
                )
                print("画像ベクトルで検索しました")
            else:
                raise ValueError("無効な検索方法です")

            results = cur.fetchall()  # tuple[image_id, file_name, score, method]

            # LOBオブジェクトを文字列に変換
            processed_results = []
            for row in results:
                (
                    image_id,
                    file_name,
                    score,
                    method,
                ) = row
                processed_results.append(
                    (
                        image_id,
                        file_name,
                        score,
                        method,
                    )
                )

    print(f"検索結果: {len(processed_results)}件")
    return processed_results


# ? ========== ========== ========== ========== ==========
# ? グローバル関数のヘルパー関数
# ? ========== ========== ========== ========== ==========

login("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")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stabilityai/japanese-stable-clip-vit-l-16"
MODEL = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_PATH)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def tokenize(
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    global TOKENIZER

    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = TOKENIZER(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[TOKENIZER.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )


def compute_text_embeddings(text):
    """
    テキストの埋め込みベクトルを計算する関数。

    Args:
        text (str or List[str]): 埋め込みを計算するテキスト。単一の文字列または文字列のリスト。

    Returns:
        torch.Tensor: 正規化されたテキストの埋め込みベクトル。

    処理の流れ:
    1. 入力が単一の文字列の場合、リストに変換。
    2. テキストをトークン化。
    3. モデルを使用してテキスト特徴量を抽出。
    4. 特徴量ベクトルを正規化。
    5. 不要なメモリを解放。
    6. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    global MODEL
    global DEVICE

    if isinstance(text, str):
        text = [text]
    text = tokenize(texts=text)
    text_features = MODEL.get_text_features(**text.to(DEVICE))
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text
    return text_features.cpu().detach()


MODEL_ENET = keras.applications.EfficientNetB0(include_top=False, pooling="avg")


# ? search_imagesのヘルパー関数
# todo global変数あり
def compute_image_embeddings(image):
    global MODEL_ENET

    # GradioのPIL画像をJPEG形式のバイト列に変換
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()

    # JPEGバイトデータをTensorFlowで読み込み
    img = tf.image.decode_jpeg(byte_data, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = keras.applications.efficientnet.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # バッチ次元を追加

    # ベクトル化と正規化
    vec = MODEL_ENET.predict(img)
    vec = vec.astype("float32")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec2str(vec)

    return vec[0]


def vec2str(vecs) -> list:
    size = int(vecs[0].shape[0])
    vecs_s = []
    for vec in vecs:
        s = "["
        for idx, scala in enumerate(vec):
            s += str(scala)
            if idx != size - 1:
                s += ","
        s += "]"
        vecs_s.append(s)
    return vecs_s


load_sql = """
SELECT image_id, file_name FROM images
ORDER BY image_id
DESC LIMIT %s
OFFSET %s;
"""


# ? load_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# * 最新の画像をoffsetからlimit分取得する
# * タプル(image_id, file_name)をリストで返す
def get_latest_images(limit=IMAGES_PER_PAGE, offset=0) -> list[tuple[int, str]]:
    global IMAGES_PER_PAGE
    global DSN
    global load_sql
    conn = None
    cur = None
    results = None
    processed_results = None
    image_id = None
    file_name = None
    row = None

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:

            cur.execute(
                load_sql,
                (limit, offset),
            )

            results = cur.fetchall()

            # LOBオブジェクトを文字列に変換
            processed_results = []
            for row in results:
                image_id, file_name = row
                processed_results.append(
                    (
                        image_id,
                        file_name,
                    )
                )

    return processed_results


# ! ========== ========== ========== ========== ==========
# ! Gradioアプリ
# ! ========== ========== ========== ========== ==========

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

    # ? ========== ========== ========== ========== ==========
    # ? Gradioのローカル関数
    # ? ========== ========== ========== ========== ==========

    # ? get_total_pagesのヘルパー関数
    # todo DB操作あり
    # todo global変数あり
    # * DBに登録されている画像数を取得する
    def get_total_image_count() -> int:
        global DSN
        conn = None
        cur = None
        count = None

        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM images;")
                count = cur.fetchone()[0]
        return count

    # ? change_pageとsearch_wrapperとload_initial_galleryのヘルパー関数
    # todo global変数あり
    # * ( DBに登録されている画像数 ÷ ギャラリーに一度に表示する画像数 ) の商 = 全ページ数
    def get_total_pages() -> int:
        global IMAGES_PER_PAGE
        total_images = None

        total_images = get_total_image_count()  # 500
        # (500 + 16 - 1) = 515 ÷ 16 = 32 ... 3
        # return 32
        return (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE

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

    # ? ========== ========== ========== ========== ==========
    # ? イベントハンドラ
    # ? ========== ========== ========== ========== ==========

    # ? イベントハンドラ：Gradioアプリ(demo)起動時
    # * input :
    # * output : (6) gallery,image_info_state,prev_button,next_button,current_page,page_info,
    def load_initial_gallery():
        images = None
        image_info = None
        total_pages = None
        page_info_text = None

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

    # ? イベントハンドラ：「検索」ボタン押下時
    # * input : (3) text_input,image_input,search_method
    # * output : (9) gallery,image_info_state,text_input,image_input,search_button,gallery,prev_button,next_button,page_info
    def search_wrapper(text_query, image_query, search_method):
        results = None
        vector_results = None
        vector_info = None
        image_ids = None
        image_data_dict = None
        image_id = None
        file_name = None
        score = None
        method = None
        image_data = None
        img = None
        info = None
        caption = None
        gallery_images = None
        image_info = None
        page_info_text = None
        images = None
        result = None

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
                gr.update(
                    interactive=(search_method == "画像ベクトル検索")
                ),  # image_input
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

    # ? イベントハンドラ：「前」ボタン押下時
    # * input : (1) current_page
    # * output : (6) gallery,image_info_state,prev_button,next_button,current_page,page_info
    def prev_page(current_page):
        return change_page("prev", current_page)

    # ? イベントハンドラ：「次」ボタン押下時
    # * input : (1) current_page
    # * output : (6) gallery,image_info_state,prev_button,next_button,current_page,page_info
    def next_page(current_page):
        return change_page("next", current_page)

    # ? ========== ========== ========== ========== ==========
    # ? イベントリスナ
    # ? ========== ========== ========== ========== ==========

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

# ! ========== ========== ========== ========== ==========
# ! エントリーポイント
# ! ========== ========== ========== ========== ==========

if __name__ == "__main__":
    try:
        demo.queue()
        demo.launch(debug=True, share=True)
    except Exception as e:
        print(e)
        demo.close()
