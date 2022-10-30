import pandas

def main():
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
        word_pairs.append(line.split('\t'))
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main()
