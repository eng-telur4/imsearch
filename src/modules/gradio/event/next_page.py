from modules.gradio.app.change_page import change_page


# ? イベントハンドラ：「次」ボタン押下時
# * input : (1) current_page
# * output : (6) gallery,image_info_state,prev_button,next_button,current_page,page_info
def next_page(current_page):
    return change_page("next", current_page)