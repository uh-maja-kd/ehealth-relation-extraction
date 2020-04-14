import torch
from utils import to_categorical, build_bert_tokenizer

# pylint: disable=no-member, not-callable

RELATIONS = [
    "none",
    "subject",
    "target",
    "in-place",
    "in-time",
    "in-context",
    "arg",
    "domain",
    "has-property",
    "part-of",
    "is-a",
    "same-as",
    "causes",
    "entails",
]


class BertRelationExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, collection):
        self.collection = collection
        self.bert_tokenizer = build_bert_tokenizer()

    def __len__(self):
        return len(self.collection.sentences)

    def __getitem__(self, index):
        sentence = self.collection.sentences[index]
        batch = self.build_batch(sentence)

        batch_input, batch_output = tuple(zip(*tuple(batch)))
        sents, starts_1, starts_2 = tuple(zip(*tuple(batch_input)))

        sents = torch.tensor(sents)
        starts_1 = to_categorical(torch.tensor(starts_1), sents.size()[-1])
        starts_1.unsqueeze_(-2)
        starts_2 = to_categorical(torch.tensor(starts_2), sents.size()[-1])
        starts_2.unsqueeze_(-2)

        batch_output = torch.tensor(batch_output)

        return (sents, starts_1, starts_2), batch_output

    def __iter__(self):
        idxs = list(range(len(self)))
        for i in idxs:
            yield self[i]

    def build_batch(self, sentence):
        for kp1 in sentence.keyphrases:
            for kp2 in sentence.keyphrases:
                if kp1.id == kp2.id:
                    continue

                yield self.build_input(
                    sentence.text,
                    (kp1.spans[0][0], kp1.spans[-1][1]),
                    (kp2.spans[0][0], kp2.spans[-1][1]),
                ), self.build_output(sentence, kp1.id, kp2.id)

    def build_input(self, text, span_1, span_2):
        t1, t2, t3, t4 = sorted(
            (
                (span_1[0], "[START1]"),
                (span_1[1], "[END1]"),
                (span_2[0], "[START2]"),
                (span_2[1], "[END2]"),
            ),
            key=lambda i: i[0],
        )
        text = (
            text[: t1[0]]
            + t1[1]
            + text[t1[0] : t2[0]]
            + t2[1]
            + text[t2[0] : t3[0]]
            + t3[1]
            + text[t3[0] : t4[0]]
            + t4[1]
            + text[t4[0] :]
        )
        tokens = self.bert_tokenizer.tokenize(text)
        start_1, start_2 = tokens.index(t1[1]), tokens.index(t3[1])
        ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        return ids, start_1, start_2

    def build_output(self, sentence, kp1, kp2):
        relations = [rel.label for rel in sentence.find_relations(kp1, kp2)]
        output = [0] * len(RELATIONS)
        for i, rel in enumerate(RELATIONS):
            if rel in relations:
                output[i] = 1
        return output

