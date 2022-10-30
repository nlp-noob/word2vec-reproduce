import pandas
import torch

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
        a_pair[3] = float(str(a_pair[3][4:]))
        word_pairs.append(a_pair)
    return word_pairs

def get_embeddings():



def main():
    folder = "../weights/cbow_Win5_BAT40_DIM300_Mul95.0/"
    model = "191epochs_model.pt"
    device = torch.device("cpu")
    word_pairs = get_test_data()
    pass


if __name__=="__main__":
    main()
