import gradio as gr


def greet(name, intensity):
    return f"Hello, {name}!\n" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()

# Interface : 関数をUIでラップするためのクラス
#     fn      : ラップする関数
#     inputs  : 入力コンポーネント
#     outputs : 出力コンポーネント
