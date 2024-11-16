from modules.gradio.app.change_page import change_page


# ? イベントハンドラ：「前」ボタン押下時
# * input : (1) current_page
# * output : (6) gallery,image_info_state,prev_button,next_button,current_page,page_info
def prev_page(current_page):
    return change_page("prev", current_page)
