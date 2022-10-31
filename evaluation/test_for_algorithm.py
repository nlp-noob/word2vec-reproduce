import numpy as np
import pandas as pd
import torch
import sys

from sklearn.manifold import TSNE
import plotly.graph_objects as go

sys.path.append("../")
folder = "weights/skipgram_WikiText2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(f"../{folder}/model.pt", map_location=device)
vocab = torch.load(f"../{folder}/vocab.pt")
# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()
# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape
# get embeddings
embeddings_df = pd.DataFrame(embeddings)


emb1 = embeddings[vocab["king"]]
emb2 = embeddings[vocab["man"]]
emb3 = embeddings[vocab["woman"]]

emb4 = emb1 - emb2 + emb3
emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
emb4 = emb4 / emb4_norm

emb4 = np.reshape(emb4, (len(emb4), 1))
dists = np.matmul(embeddings_norm, emb4).flatten()

top5 = np.argsort(-dists)[:5]

for word_id in top5:
    print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))
