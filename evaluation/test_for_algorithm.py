import numpy as np
import pandas as pd
import torch
import sys
import os

from sklearn.manifold import TSNE
import plotly.graph_objects as go

def test():
    folder_list = os.listdir("../weights")
    best_epoch_list = [13, 40, 5, 28, 9, 3]
    epoch_index = 0
    for folder in folder_list:
        folder = "../weights/" + folder
        epoch = best_epoch_list[epoch_index]
        epoch_index += 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
        model = torch.load(f"{folder}/{epoch}epochs_model.pt", map_location=device)
        vocab = torch.load(f"{folder}/vocab.pt")
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
        
        
        emb1 = embeddings[vocab["black"]]
        emb2 = embeddings[vocab["green"]]
        emb3 = embeddings[vocab["blue"]]
        
        emb4 = emb1 - emb2 + emb3
        emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
        emb4 = emb4 / emb4_norm
        
        emb4 = np.reshape(emb4, (len(emb4), 1))
        dists = np.matmul(embeddings_norm, emb4).flatten()
        
        top5 = np.argsort(-dists)[:5]
        print("===========================")
        print("In model {}".format(folder)) 
        for word_id in top5:
            print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))
        print("===========================")

if __name__=="__main__":
    test()
