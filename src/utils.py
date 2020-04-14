import torch
from transformers import BertTokenizer, BertModel

# pylint: disable=no-member, not-callable


def to_categorical(x, num_classes):
    assert x.dim() == 1 and num_classes > max(x)
    n = x.size()[0]
    categorical = torch.zeros(n, num_classes)
    categorical[torch.arange(n), x] = 1
    return categorical


def build_bert_tokenizer():
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[START1]", "[START2]", "[END1]", "[END2]"]}
    )

    return bert_tokenizer


def build_bert_model_builder():
    bert_tokenizer = build_bert_tokenizer()
    vocab_size = len(bert_tokenizer)

    def build_bert_model() -> BertModel:
        bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        bert_model.tie_weights()  # don't know what this is for
        bert_model.resize_token_embeddings(vocab_size)
        return bert_model

    return build_bert_model


build_bert_model = build_bert_model_builder()
