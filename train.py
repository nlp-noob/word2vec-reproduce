# 代码开始进行训练的入口
# 其中的过程：
import argparse
import random
import yaml
import torch
import torch.optim as optim
import os
import dataloader
import model
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.data import get_tokenizer
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


# 创建Trainer类
class Trainer:
    def __init__(
            self,
            iteror,
            config,
            vocab):        
        self.config = config
        self.vocab_size = len(vocab.get_itos())
        self.model = None
        self.init_model()
        self.epochs = config["epochs"]
        self.iteror = iteror
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer, self.epochs, True)
        self.lr = self.lr_scheduler.get_last_lr()[0]

    def init_model(self):    
        if(self.config["model"]=="cbow"):
            train_model = model.CBOW_MODEL
        elif(self.config["model"]=="skipgram"):
            train_model = model.SkipGram_MODEL
        self.model = train_model(self.vocab_size, self.config)

    def get_lr_scheduler(self, optimizer, total_epochs: int, verbose: bool = True):
        # lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
        if(self.config["scheduler"]=="MultiplicativeLR"):
            lr_multiplicative = lambda epoch: self.config["Mul"]
            lr_scheduler = MultiplicativeLR(optimizer, lr_multiplicative, verbose=verbose)
        # lr_scheduler = CosineAnnealingLR(optimizer, 100)
        return lr_scheduler

    def train_an_epoch(self, batches, labels, epoch):
        sum_loss = 0
        self.model.train()
        for batch_index in range(len(batches)):
            train_inputs = torch.tensor(batches[batch_index]).to(self.device)
            train_label = torch.tensor(labels[batch_index]).to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(train_inputs)
            loss = self.criterion(outputs, train_label)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss
        mean_loss = sum_loss/(len(batches))
        print("in epoch {} ,the mean train loss is {}".format(epoch, mean_loss))
        self.lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()
        return mean_loss

    def save_model(self, epoch):
        model_path = os.path.join(self.config["model_dir"], "{}epochs_model.pt".format(epoch))
        torch.save(self.model, model_path)

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
    if(config["shuffle"]==1):
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

def skipgram_batches_generator(config,iterator,vocab):
    # skipgram的batches其实就是CBOW的labels
    # labels就是batches
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
                a_label.append(indexs[c_index])
            for piece_x in a_batch_piece:
                a_batch.append(piece_x)
        num_sentence += 1
        if(num_sentence>=batch_size):
            batches.append(a_batch)
            labels.append(a_label)
            a_batch = []
            a_label = []
            num_sentence = 0
    skipgram_batches = labels
    skipgram_labels = batches
    return skipgram_batches, skipgram_labels
    
def get_batches_list(config, iterator, vocab):
    if(config["model"]=="cbow"):
        batches, labels = CBOW_batches_generator(config, iterator, vocab)
    if(config["model"]=="skipgram"):
        batches, labels = skipgram_batches_generator(config, iterator, vocab)
    return batches, labels

# 验证一个epoch的模型
def valid_an_epoch(config, valid_model, iteror, vocab, criterion, epoch):
    valid_batches, valid_labels = get_batches_list(config, iteror, vocab)
    device = torch.device(config["device"])
    sum_loss = 0
    valid_model.eval()
    with torch.no_grad():
        for batch_index in range(len(valid_batches)):
            valid_inputs = torch.tensor(valid_batches[batch_index]).to(device)
            valid_label = torch.tensor(valid_labels[batch_index]).to(device)
            outputs = valid_model(valid_inputs)
            loss = criterion(outputs, valid_label)
            sum_loss += loss
        mean_loss = sum_loss/(len(valid_batches))
        print("in epoch {} ,the mean valid loss is {}".format(epoch, mean_loss))
    return mean_loss

def train(config, train_iteror, valid_iteror, vocab):
    # 创建训练器对象
    trainer = Trainer(train_iteror, config, vocab)
    writer = SummaryWriter(os.path.join(config["model_dir"], "log"))
    for epoch in range(trainer.epochs):
        # 开始迭代前验证
        valid_loss = valid_an_epoch(trainer.config, trainer.model, valid_iteror, vocab, trainer.criterion, epoch)
        trainer.save_model(epoch)

        # 训练,并计算训练的loss
        train_batches, train_labels = get_batches_list(config, train_iteror, vocab)
        training_loss = trainer.train_an_epoch(train_batches, train_labels, epoch)
        
        writer.add_scalar("training_loss", float(training_loss), global_step=(epoch))
        writer.add_scalar("learning_rate", float(trainer.lr), global_step=(epoch))
        writer.add_scalar("valid_loss", float(valid_loss), global_step=(epoch))

def format_model_dir(config):
    if(config["scheduler"]=="MultiplicativeLR"):
        modeldir = "./weights/{}_Win{}_BAT{}_DIM{}_Mul{}".format(config["model"], 
                                                            config["win_size"], 
                                                            config["BATCH_SIZE"],
                                                            config["EMBED_DIMENSION"],
                                                            config["Mul"]) 
    else:
        modeldir = "./weights/{}_Win{}_BAT{}_DIM{}".format(config["model"], 
                                                            config["win_size"], 
                                                            config["BATCH_SIZE"],
                                                            config["EMBED_DIMENSION"]) 
    return modeldir

def main():
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream) 
    train_iteror, valid_iteror, vocab = dataloader.get_data_and_vocab(config)
    config["model_dir"] = format_model_dir(config) 
    os.makedirs(config["model_dir"], exist_ok=True)
    torch.save(vocab, os.path.join(config["model_dir"], "vocab.pt"))
    # 开始训练
    train(config, train_iteror, valid_iteror, vocab)

if __name__=="__main__":
    main()

