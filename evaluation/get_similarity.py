import os
import numpy as np
import pandas
import torch
from torch.utils.tensorboard import SummaryWriter

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
    for i in  range(len(f_epoch)):
        file_path = weight_path + str(i) + "epochs_model.pt"
        # 在这里进行读入的时候需要注意的是，需要将定义模型的model文件放到一个文件夹中
        my_model = torch.load(file_path, map_location=torch.device("cpu"))
        embeddings = list(my_model.parameters())[0]
        embeddings_list.append(embeddings)
    return embeddings_list

def cosine_vec(a_vec, b_vec):
    a_vec = a_vec.cpu().detach().numpy()
    b_vec = b_vec.cpu().detach().numpy()
    num = float(np.dot(a_vec, b_vec))  # 向量点乘
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def main():
    # original code path
    # folder = "../../LearningWord2vec/word2vec-pytorch-main/weights/cbow_Win5_WikiText2/"
    folder = "../weights/WikiText2_cbow_Win5_BAT96_DIM500_Mul99/"
    word_pairs = get_test_data()
    embeddings_list = get_embeddings(folder)
    vocab = torch.load(folder+"vocab.pt")
    writer = SummaryWriter(folder+"log")
    corr_list = []
    for index in range(len(embeddings_list)):
        similarity_list = []
        for word_pair in word_pairs:
            a_vec_index = vocab[word_pair[1]]
            b_vec_index = vocab[word_pair[2]]
            if(a_vec_index==0 or b_vec_index==0):
                continue
            a_vec = embeddings_list[index][a_vec_index]
            b_vec = embeddings_list[index][b_vec_index]
            sim = cosine_vec(a_vec, b_vec)
            similarity_list.append((sim, word_pair[3]))
        df = pandas.DataFrame(similarity_list)
        corr = df.corr()[0][1]
        corr_list.append(corr)
        if(index%10==0):
            print("the {} corr_similarity is {}".format(index, corr))
        epoch = index
        writer.add_scalar("corr_similarity", float(corr), global_step=(epoch))
    print("the max similarity is {}, epoch{}".format(np.max(corr_list), np.where(corr_list==np.max(corr_list))))
    print("the random embeddings similarity is {}".format(corr_list[0]))

if __name__=="__main__":
    main()
