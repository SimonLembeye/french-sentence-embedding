import os
import json

import numpy as np
from typing import List

import torch
from torch import nn
from transformers import CamembertModel, CamembertTokenizer


class CamemBERTEmbed(nn.Module):
    """CamemBERT model to generate token embeddings.
    Each token is mapped to an output vector from CamemBERT.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        do_lower_case: bool = True,
    ):
        super(CamemBERTEmbed, self).__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case

        if max_seq_length > 511:
            print(
                "CamemBERT only allows a max_seq_length of 511 (514 with special tokens). Value will be set to 511"
            )
            max_seq_length = 511
        self.max_seq_length = max_seq_length

        self.camembert = CamembertModel.from_pretrained(model_name_or_path)
        self.tokenizer = CamembertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=do_lower_case
        )
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token]
        )[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token]
        )[0]

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        # CamemBERT does not use token_type_ids
        input_ids = features["input_ids"]
        attention_mask = features["input_mask"]

        output_tokens = self.camembert(
            input_ids=input_ids.view(-1, 131),
            token_type_ids=None,
            attention_mask=attention_mask.view(-1, 131),
        )[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update(
            {
                "token_embeddings": output_tokens,
                "cls_token_embeddings": cls_tokens,
                "input_mask": features["input_mask"],
            }
        )
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.camembert.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length)

        tokens = tokens[:pad_seq_length]
        input_ids = (
            [self.cls_token_id] + tokens + [self.sep_token_id] + [self.sep_token_id]
        )
        sentence_length = len(input_ids)

        pad_seq_length += 3  ##Add Space for CLS + SEP + SEP token

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. BERT: Pad to the right
        padding = [0] * (pad_seq_length - len(input_ids))
        input_ids += padding

        input_mask += padding

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return {
            "input_ids": torch.tensor([np.asarray(input_ids, dtype=np.int64)]).to(device),
            "input_mask": torch.tensor([np.asarray(input_mask, dtype=np.int64)]).to(device),
            "sentence_lengths": np.asarray(sentence_length, dtype=np.int64),
        }

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.camembert.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(
            os.path.join(output_path, "sentence_camembert_config.json"), "w"
        ) as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "sentence_camembert_config.json")) as fIn:
            config = json.load(fIn)
        return CamemBERTEmbed(model_name_or_path=input_path, **config)
