import torch.nn as nn

class CBOW_MODEL(nn.Module):
    def __init__(self, vocab_size: int, config):
        super(CBOW_MODEL, self).__init__()
        self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config["EMBED_DIMENSION"],
                max_norm = config["EMBED_MAX_NORM"]
                )
        self.linear = nn.Linear(
                in_features=config["EMBED_DIMENSION"],
                out_features=vocab_size,
                )
    def forward(self, inputs_):
        x=self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class SkipGram_MODEL(nn.Module):
    def __init__(self, vocab_size: int, config):
        super(SkipGram_MODEL, self).__init__()
        self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config["EMBED_DIMENSION"],
                max_norm = config["EMBED_MAX_NORM"]
                )
        self.linear = nn.Linear(
                in_features=config["EMBED_DIMENSION"],
                out_features=vocab_size,
                )
    def forward(self, inputs_):
        x=self.embeddings(inputs_)
        x = self.linear(x)
        return x
