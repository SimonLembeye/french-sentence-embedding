from torch import nn

from models.camembert_embed import CamemBERTEmbed
from utils.pooling import Pooling


class SentenceEmbedder(nn.Module):
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
        super(SentenceEmbedder, self).__init__()
        self.word_embedding_model = CamemBERTEmbed(
            model_name_or_path,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
        )
        self.pooling_model = Pooling(
            self.word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
        self.max_seq_length = max_seq_length

    def forward(self, features):
        sentence_embedding_features = self.word_embedding_model(features)
        return self.pooling_model(sentence_embedding_features)["sentence_embedding"]
