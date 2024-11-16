from modules.settings import IMAGES_PER_PAGE
from modules.db.get_total_image_count import get_total_image_count


# ? change_pageとsearch_wrapperとload_initial_galleryのヘルパー関数
# todo global変数あり
# * ( DBに登録されている画像数 ÷ ギャラリーに一度に表示する画像数 ) の商 = 全ページ数
def get_total_pages() -> int:
    global IMAGES_PER_PAGE

    total_images = get_total_image_count()  # 500
    # (500 + 16 - 1) = 515 ÷ 16 = 32 ... 3
    # return 32
    return (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
