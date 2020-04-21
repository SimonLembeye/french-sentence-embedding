import torch
from torch import nn

from models.sentence_embedder import SentenceEmbedder


class SiameseCamemBERT(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        do_lower_case: bool = True,
        word_embedding_dimension: int = 768,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
    ):
        super(SiameseCamemBERT, self).__init__()
        self.sentence_embedder = SentenceEmbedder(
            model_name_or_path,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            word_embedding_dimension=word_embedding_dimension,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens=pooling_mode_mean_sqrt_len_tokens,
        )
        self.linear = torch.nn.Linear(3 * word_embedding_dimension, 3)

    def forward(self, x):
        u = self.sentence_embedder(x["sentence1"])
        v = self.sentence_embedder(x["sentence2"])
        d = torch.abs(u - v)
        z = torch.cat((u, v, d), 1)
        return self.linear(z)
