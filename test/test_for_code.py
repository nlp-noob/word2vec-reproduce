# 这是一个测试产生的随机batch是否与原来使用的dataloader相同的测试脚本
from torchtext.datasets import WikiText2
from torchtext.datasets import WikiText103
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from functools import partial
import yaml
import random

SHUFFLE = 0

def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)
        if len(text_tokens_ids) < 5 * 2 + 1:
            continue
        text_tokens_ids = text_tokens_ids[:256]
        for idx in range(len(text_tokens_ids) - 5*2):
            token_id_sequence = text_tokens_ids[idx : (idx + 5 * 2 + 1)]
            output = token_id_sequence.pop(5)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)
    return batch_input, batch_output

# 生成由整数组成的数组，每一个整数元素代表坐标对应的词
def sentences_to_index_lists(config, iterator, vocab):
    sentences = list(iterator)
    tokenizer = get_tokenizer("basic_english")
    # 转换成indexs的sentence列表集合
    index_lists = []
    for sentence in sentences:
        sentence = tokenizer(sentence)
        index_list = []
        for word in sentence:
            index_list.append(vocab[word])
        index_lists.append(index_list)
    # 根据设置决定是否打乱数据集
    if(SHUFFLE==1):
        random.shuffle(index_lists)
    return index_lists

# 生成batch_size内的batches
def CBOW_batches_generator(config, iterator, vocab):
    window_size = config["win_size"]
    max_sequence = config["MAX_SEQUENCE_LENGTH"]
    batch_size = config["BATCH_SIZE"]
    index_lists = sentences_to_index_lists(config, iterator, vocab)
    # 此处进行window和label的抽取
    batches = []
    a_batch = []
    labels = []
    a_label = []
    window_len = window_size * 2 + 1
    num_sentence = 0
    for indexs in index_lists:
        if(len(indexs)>max_sequence):
            continue
        if(len(indexs)<(window_len)):
            continue
        for i in range(len(indexs)-(window_size*2)):
            c_index = i+window_size # 中心词的坐标
            a_batch_piece = [0 for z in range(0, 2 * window_size)] 
            for j in range(window_size):
                a_batch_piece[window_size - j - 1] = indexs[c_index - j - 1]
                a_batch_piece[window_size + j] = indexs[c_index + j + 1] 
            a_label.append(indexs[c_index])
            a_batch.append(a_batch_piece)
        num_sentence += 1
        if(num_sentence>=batch_size):
            batches.append(a_batch)
            labels.append(a_label)
            a_batch = []
            a_label = []
            num_sentence = 0
    return batches, labels


def build_vocab(data_iter, tokenizer, config):
    """Builds vocabulary from iterator"""
    vocab = build_vocab_from_iterator(
                            map(tokenizer, data_iter),
                            specials=["<unk>"],
                            min_freq=config['MIN_WORD_FREQUENCY'],)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def get_data_and_vocab(config):
    dataset = WikiText103
    train_iteror = dataset("../"+config["data_dir"], split="train") 
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_iteror, tokenizer, config)
    return train_iteror, vocab

def test(): 
    random.seed(42)
    with open("../config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    train_iteror, vocab = get_data_and_vocab(config)
    random.seed(42)
    tokenizer = get_tokenizer("basic_english")
    # get total_words_num
    totalword = 0
    for sentence in list(train_iteror):
        sentence = tokenizer(sentence)
        totalword += len(sentence)
    print(f"the total word in corpus is:{totalword}")
    batches, labels = CBOW_batches_generator(config, train_iteror, vocab)
    
    text_pipline = lambda x: vocab(tokenizer(x))
    dataloader = DataLoader(train_iteror, config["BATCH_SIZE"],
                            shuffle=SHUFFLE, 
                            collate_fn=partial(collate_cbow, text_pipeline=text_pipline))
    test_index = 1
    for i, batch_data in enumerate(dataloader, 1):
        if (i==test_index):
            print(batch_data[0][1])
            print(batch_data[1][1])
            break
    print(batches[test_index+1][1])
    print(labels[test_index+1][1])


if __name__=="__main__":
    test()
