import re


from modules.gradio.atoi import atoi


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]
