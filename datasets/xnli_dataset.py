import pandas as pd

import torch
from torch.utils.data import Dataset


class XNLIDataset(Dataset):
    def __init__(
        self, sentence_embedder_model, tsv_file_path="french_XNLI/multinli.train.fr.tsv"
    ):
        df = pd.read_csv(tsv_file_path, sep="\t")[["premise", "hypo", "label"]]
        df = df.drop(df[df.isnull().values].index)
        self.data_file = df
        self.sentence_embedder_model = sentence_embedder_model

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data_file.iloc[idx]

        sentence_token1 = self.sentence_embedder_model.word_embedding_model.tokenize(
            row["premise"]
        )
        features1 = self.sentence_embedder_model.word_embedding_model.get_sentence_features(
            sentence_token1, self.sentence_embedder_model.max_seq_length
        )

        sentence_token2 = self.sentence_embedder_model.word_embedding_model.tokenize(
            row["hypo"]
        )
        features2 = self.sentence_embedder_model.word_embedding_model.get_sentence_features(
            sentence_token2, self.sentence_embedder_model.max_seq_length
        )

        sample = {"sentence1": features1, "sentence2": features2, "label": row["label"]}

        return sample
