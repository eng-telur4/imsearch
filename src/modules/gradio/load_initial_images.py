import io
from PIL import Image
from typing import Union
from PIL.ImageFile import ImageFile
from modules.settings import IMAGES_PER_PAGE
from modules.db.get_latest_images import get_latest_images
from modules.db.get_multiple_image_data import get_multiple_image_data

t_vector_distance = Union[float, str]

t_images = list[ImageFile]
t_image_info = list[dict[str, t_vector_distance]]


# ? search_wrapperとchange_pageとload_initial_galleryのヘルパー関数
# todo global変数あり
def load_initial_images(page=1) -> tuple[t_images, t_image_info]:
    global IMAGES_PER_PAGE

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
