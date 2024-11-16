import re
import os
import io
import oci
import ftfy
import html
import json
import torch
import spaces
import cohere
import oracledb
import gradio as gr
from PIL import Image
from typing import Union, List
from oci.config import from_file
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from oci.generative_ai_inference import GenerativeAiInferenceClient
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode

_ = load_dotenv(find_dotenv())

# ! load_initial_images()
# ! get_total_pages()
IMAGES_PER_PAGE = 16

# データベース接続情報
# todo DB操作あり
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# Japanese Stable CLIPモデルのロード
login(os.getenv("HF_TOKEN"))
# ! compute_text_embeddings()
# ! compute_image_embeddings()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "stabilityai/japanese-stable-clip-vit-l-16"
# ! compute_text_embeddings()
# ! compute_image_embeddings()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
# ! tokenize()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ! compute_image_embeddings()
processor = AutoImageProcessor.from_pretrained(model_path)

# OCI設定
# todo unused
CONFIG_PROFILE = "DEFAULT"
config = from_file("~/.oci/config", CONFIG_PROFILE)
compartment_id = os.getenv("OCI_COMPARTMENT_ID")
model_id = "cohere.embed-multilingual-v3.0"
generative_ai_inference_client = GenerativeAiInferenceClient(
    config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10, 240)
)

# Cohere設定
# todo unused
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


# ! ========== ========== ========== ========== ==========
# ! グローバル関数
# ! ========== ========== ========== ========== ==========

# ? ========== ========== ========== ========== ==========
# ? グローバル関数
# ? ========== ========== ========== ========== ==========


# ? search_wrapperとchange_pageとload_initial_galleryのヘルパー関数
# todo global変数あり
def load_initial_images(page=1):
    global IMAGES_PER_PAGE
    offset = None
    results = None
    image_ids = None
    image_data_dict = None
    images = None
    image_info = None
    image_id = None
    file_name = None
    image_data = None

    offset = (page - 1) * IMAGES_PER_PAGE
    results = get_latest_images(limit=IMAGES_PER_PAGE, offset=offset)

    # 画像IDのリストを作成
    image_ids = [result[0] for result in results]

    # 一度にすべての画像データを取得
    image_data_dict = get_multiple_image_data(image_ids)

    images = []
    image_info = []
    for image_id, file_name in results:
        image_data = image_data_dict.get(image_id)
        if image_data:
            images.append(Image.open(io.BytesIO(image_data)))
            image_info.append(
                {
                    "file_name": file_name,
                    "vector_distance": "N/A",
                }
            )

    return images, image_info


# ? search_wrapperとload_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
# 引数のimage_idからimage_dataを取得する
def get_multiple_image_data(image_ids):
    global username
    global password
    global dsn
    conn = None
    cur = None
    placeholders = None
    query = None
    bind_vars = None
    image_data_dict = None

    if not image_ids:
        return {}  # 画像IDが空の場合は空の辞書を返す

    with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
        with conn.cur() as cur:
            placeholders = ",".join([":" + str(i + 1) for i in range(len(image_ids))])
            query = f"SELECT image_id, image_data FROM IMAGES WHERE image_id IN ({placeholders});"

            # バインド変数を辞書形式で渡す
            bind_vars = {str(i + 1): image_id for i, image_id in enumerate(image_ids)}

            cur.execute(query, bind_vars)

            image_data_dict = {row[0]: row[1].read() for row in cur}

    return image_data_dict


search_sql = """
SELECT i.image_id, i.file_name, i.generation_prompt,
    id.description as combined_description,
    cie.embedding <#> :query_embedding as vector_distance,
    'vector' as method
FROM CURRENT_IMAGE_EMBEDDINGS cie
JOIN IMAGES i ON cie.image_id = i.image_id
LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
ORDER BY vector_distance
FETCH FIRST :limit ROWS ONLY
"""

# search_sql = f"""
# SELECT image_id, file_name,
#     1 - (embedding <#> {query_embedding}) as cosine_similarity,
#     'vector' as method
# FROM IMAGES
# ORDER BY cosine_similarity
# DESC LIMIT {limit};
# """


# ? search_wrapperのヘルパー関数
# todo DB操作あり
# todo global変数あり
@spaces.GPU(duration=60)
def search_images(query, search_method, limit=16):
    global username
    global password
    global dsn
    global search_sql
    conn = None
    cur = None
    embedding_json = None
    results = None
    processed_results = None
    image_id = None
    file_name = None
    score = None
    method = None
    row = None

    with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
        with conn.cur() as cur:

            if search_method == "自然言語ベクトル検索":
                embedding_json = json.dumps(
                    compute_text_embeddings(query).tolist()[0],
                )
                cur.execute(
                    search_sql,
                    {"query_embedding": embedding_json, "limit": limit},
                )
            elif search_method == "画像ベクトル検索":
                embedding_json = json.dumps(
                    compute_image_embeddings(query).tolist()[0],
                )
                cur.execute(
                    search_sql,
                    {"query_embedding": embedding_json, "limit": limit},
                )
            else:
                raise ValueError("無効な検索方法です")

            results = cur.fetchall()

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


# ? tokenizeのヘルパー関数
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


# ? tokenizeのヘルパー関数
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ? compute_text_embeddingsのヘルパー関数
# todo global変数あり
def tokenize(texts: Union[str, List[str]], max_seq_len: int = 77):
    global tokenizer
    inputs = None
    input_ids = None
    attention_mask = None
    position_ids = None

    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )


# ? search_imagesのヘルパー関数
# todo global変数あり
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
    global model
    global device
    text_features = None

    if isinstance(text, str):
        text = [text]
    text = tokenize(texts=text)
    text_features = model.get_text_features(**text.to(device))
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text
    return text_features.cpu().detach()


# ? search_imagesのヘルパー関数
# todo global変数あり
def compute_image_embeddings(image):
    """
    画像の埋め込みベクトルを計算する関数。

    Args:
        image (PIL.Image.Image): 埋め込みを計算する画像。

    Returns:
        torch.Tensor: 正規化された画像の埋め込みベクトル。

    処理の流れ:
    1. 画像をモデルの入力形式に変換し、デバイスに移動。
    2. 勾配計算を無効化して、モデルを使用して画像特徴量を抽出。
    3. 特徴量ベクトルを正規化。
    4. 不要なメモリを解放。
    5. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    global processor
    global device
    global model
    image_features = None

    image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    del image
    return image_features.cpu().detach()


load_sql = """
SELECT i.image_id, i.file_name, i.generation_prompt,
    id.description AS combined_description
FROM IMAGES i
LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
ORDER BY i.upload_date DESC
OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
"""

# load_sql = f"""
# SELECT image_id, file_name
# FROM IMAGES
# ORDER BY upload_date
# OFFSET {offset} ROWS FETCH NEXT
# DESC LIMIT {limit};
# """


# ? load_initial_imagesのヘルパー関数
# todo DB操作あり
# todo global変数あり
def get_latest_images(limit=16, offset=0):
    global username
    global password
    global dsn
    global load_sql
    conn = None
    cur = None
    results = None
    processed_results = None
    image_id = None
    file_name = None
    row = None

    with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
        with conn.cur() as cur:

            cur.execute(
                load_sql,
                {"limit": limit, "offset": offset},
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

    # ? ========== ========== ========== ========== ==========
    # ? Gradioのローカル関数
    # ? ========== ========== ========== ========== ==========

    # ? get_total_pagesのヘルパー関数
    # todo DB操作あり
    # todo global変数あり
    def get_total_image_count():
        global username
        global password
        global dsn
        conn = None
        cur = None
        count = None

        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            with conn.cur() as cur:
                cur.execute("SELECT COUNT(*) FROM IMAGES;")
                count = cur.fetchone()[0]
        return count

    # ? change_pageとsearch_wrapperとload_initial_galleryのヘルパー関数
    # todo global変数あり
    def get_total_pages():
        global IMAGES_PER_PAGE
        total_images = None

        total_images = get_total_image_count()
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
        total_pages = get_total_pages()
        page_info_text = f"1 / {total_pages}"
        return (
            images,  # gallery
            image_info,  # image_info_state
            gr.update(interactive=False),  # prev_button
            gr.update(interactive=total_pages > 1),  # next_button
            1,  # current_page
            page_info_text,  # page_info
        )

    # ? イベントハンドラ：「検索方法」の項目変更時
    # * input : (1) search_method
    # * output : (1) text_input
    def update_text_input(search_method):
        if search_method == "画像ベクトル検索":
            return gr.update(value=None, interactive=False)  # text_input
        else:
            return gr.update(interactive=True)  # text_input

    # ? イベントハンドラ：「検索方法」の項目変更時
    # * input : (1) search_method
    # * output : (1) text_input
    def update_image_input(search_method):
        if search_method == "画像ベクトル検索":
            return gr.update(interactive=True)  # image_input
        else:
            return gr.update(value=None, interactive=False)  # image_input

    # ? イベントハンドラ：「検索」ボタン押下時
    # * input : (3) text_input,image_input,search_method
    # * output : (9) gallery,image_info_state,text_input,image_input,search_button,gallery,prev_button,next_button,page_info
    def search_wrapper(text_query, image_query, search_method):
        results = None
        vector_results = None
        text_results = None
        vector_info = None
        text_info = None
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

        if text_query or image_query is not None:
            results = search_images(
                text_query if text_query else image_query,
                search_method,
            )

            # 検索結果が空の場合
            if not results:
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
            text_results = []
            vector_info = []
            text_info = []

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
            gallery_images = vector_results + text_results
            image_info = vector_info + text_info

            print(f"search_method: {search_method}")

            page_info_text = ""

            # ギャラリーの更新と他のUI要素の更新
            return (
                gallery_images,  # gallery
                image_info,  # image_info_state
                gr.update(
                    interactive=(search_method != "画像ベクトル検索")
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
        else:
            # 初期表示の処理
            images, image_info = load_initial_images()
            gallery_images = [(img, None) for img in images]
            page_info_text = f"1 / {get_total_pages()}"
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
        return (
            gr.update(
                value="", interactive=(search_method != "画像ベクトル検索")
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

            return (
                info["file_name"],  # file_name
                str(vector_distance),  # distance
            )
        else:
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
        demo.launch(debug=True, share=True, server_port=8899)
    except Exception as e:
        print(e)
        demo.close()
