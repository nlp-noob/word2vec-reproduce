from torchtext.datasets import WikiText2
from torchtext.datasets import WikiText103
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def build_vocab(data_iter, tokenizer, config):
    """Builds vocabulary from iterator"""
    vocab = build_vocab_from_iterator(
                            map(tokenizer, data_iter),
                            specials=["<unk>"],
                            min_freq=config['MIN_WORD_FREQUENCY'],)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def get_data_and_vocab(config):
    if(config["dataset"]=="WikiText2"):
        dataset = WikiText2
    elif(config["dataset"]=="WikiText103"):
        dataset = WikiText103
    train_iteror = dataset(config["data_dir"], split="train") 
    valid_iteror = dataset(config["data_dir"], split="valid")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_iteror, tokenizer, config)
    return train_iteror, valid_iteror, vocab


