import gensim.downloader
import gensim
import pandas
import numpy as np
import pandas as pd
import torch
import sys
from sklearn import svm

def get_test_data():
    # pearson kendall spearman 对应着三种方法
    # pandas.DataFrame.corr(method='pearson')
    text_data_path = "test_data/wordsim353_sim_rel/wordsim353_agreed.txt"
    df = pandas.DataFrame([(1, 2), (2, 4), (3, 4)])
    print(df.corr()[0][1])
    f = open(text_data_path,"r")
    text = f.readlines()
    word_pairs = []
    for line in text:
        a_pair = []
        if(line[0]=="#"):
            continue
        a_pair = line.split('\t')
        a_pair[3] = float(str(a_pair[3][len(a_pair)-1:]))
        word_pairs.append(a_pair)
    return word_pairs

def get_embeddings(weight_path):
    f_epoch = [file_name for file_name in os.listdir(weight_path) if file_name.endswith("model.pt")]
    print("There are {} embeddings file.".format(len(f_epoch)))
    embeddings_list = []

def get_google_vecs():
    # You should modify this path to your gensim-data set in your computer
    pretrained_embeddings_path = "~/gensim-data/GoogleNews-vectors-negative300.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
    return word2vec

def cosine_vec(a_vec, b_vec):
    num = float(np.dot(a_vec, b_vec))  # 向量点乘
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def main():
    word_pairs = get_test_data()
    word2vec = get_google_vecs()
    similarity_list = []
    for word_pair in word_pairs:
        if (not(word2vec.has_index_for(word_pair[1]) and 
            word2vec.has_index_for(word_pair[2]))):
            continue
        sim = cosine_vec(word2vec[word_pair[1]], word2vec[word_pair[2]])
        similarity_list.append((sim, word_pair[3]))
    df = pandas.DataFrame(similarity_list)
    corr = df.corr()[0][1]
    print("The similarity of google word2vec model is {}".format(corr))
    

if __name__=="__main__":
    main()

