import torch
from typing import Union, List
from transformers import BatchFeature
from modules.settings import TOKENIZER
from modules.gradio.basic_clean import basic_clean
from modules.gradio.whitespace_clean import whitespace_clean


def tokenize(texts: Union[str, List[str]], max_seq_len: int = 77):
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
