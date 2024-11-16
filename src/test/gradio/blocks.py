import gradio as gr


def greet(name, intensity):
    return f"Hello, {name}!\n" * int(intensity)


with gr.Blocks() as demo:
    # name = gr.Textbox(label="name")
    # intensity = gr.Slider(label="intensity")
    # output = gr.Textbox(label="output")
    # greet_btn = gr.Button("greet")
    # greet_btn.click(fn=greet, inputs=[name, intensity], outputs=[output])
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="name")
                    intensity = gr.Slider(label="intensity")
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit", variant="primary")

        with gr.Column():
            with gr.Row():
                output = gr.Textbox(label="output")
            with gr.Row():
                flag_button = gr.Button("Flag")

    submit_button.click(fn=greet, inputs=[name, intensity], outputs=[output])


demo.launch()

# Interface : 関数をUIでラップするための高レベルクラス
#     fn      : ラップする関数
#     inputs  : 入力コンポーネント
#     outputs : 出力コンポーネント

# Blocks : Interfaceより柔軟なレイアウトとデータフローを実現するための低レベルクラス
