import torch
from utils import build_bert_model, build_bert_tokenizer, disable_grads

# pylint: disable=no-member


class BertRelationExtraction(torch.nn.Module):
    def __init__(self, relations_count, fine_tune=True):
        torch.nn.Module.__init__(self)
        self.bert_model = build_bert_model()
        if not fine_tune:
            disable_grads(self.bert_model)

        self.linear = torch.nn.Linear(
            self.bert_model.config.hidden_size * 2, relations_count
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_batch, start_1_idxs, start_2_idxs):
        bert_output = self.bert_model(input_batch)[0]
        start_1_repr = torch.matmul(start_1_idxs, bert_output).squeeze_(-2)
        start_2_repr = torch.matmul(start_2_idxs, bert_output).squeeze_(-2)

        relation_repr = torch.cat((start_1_repr, start_2_repr), -1)
        processed_relation = self.linear(relation_repr)
        output = self.sigmoid(processed_relation)

        return output

