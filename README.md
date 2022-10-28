# 代码整体框架

```python
|config.yaml # 存放各种初始化参数
	|--model_name # 用来选择相应的word2vec模型（cbow和skipgram） 
	|--dataset  # 设置相应使用的语料库 WikiText2和WikiText103
	|--data_dir # 相应的存放数据的地方
	|--data_type # 设置是使用自己的数据集还是使用导入的wiki的数据集(1,0)
  |--optimizer # 设置相应的优化器（Adam, SGD）
  |--scheduler # 设置相应的学习率衰减方法，目前只有MultiplicativeLR
  |--Mul # 假如设置了MultiplicativeLR，设置其相应的参数
  |--train_batch_size # 相应的训练数据集batch的大小
  |--val_batch_size # 相应的验证数据集的batch的大小
  |--shuffle # 是否将数据集打乱 0代表打乱，1代表不打乱
  |--learning_rate # 学习率
  |--device # 使用的训练设备 cuda或者cpu
  |--epochs # 遍历训练集模型的总迭代次数
  |--CBOW_N_WORD # 指定取词上下文窗口的大小
  |--SKIPGRAM_N_WORD 
  |--MIN_WORD_FREQUENCY # 指定进入词汇表中的词出现的最小频率
  |--MAX_SEQUENCE_LENGTH # 一个句子中最大的词语数
  |--EMBED_DIM # 词向量的维度
  |--EMBED_MAX_NORM # 使用MAX_NORM
|train.py # 主函数入口，使用python train.py --config config.yalm开始训练
|--model.py # 声明两个模型类，继承于nn.Module
      ｜--class SkipGram_Module
      ｜--class CBOW_Module 
|--dataloader.py # 用来载入数据集
|--data/ # 存放数据集的目录
|--weights/ # 存放训练模型保存数据的地方
```



# 开始训练你自己的词向量！

```
$ vim config.yaml  # 根据自己的需求调整参数
```

```
$ python train.py  # 开始训练
```

训练得到的模型保存在相应的weights文件夹中

```
$ tensorboard --logdir="weights"  # 查看所有训练模型的曲线
```





# Word2Vec核心思想

### 语义表征

我们如何用计算机语言去表征一个人类词汇的语义特征呢？

使用独热向量是最简单的方法，但是一旦使用独热向量，就会导致两个问题：

```
1.每个向量之间相互独立，无法提取相关的特性
2.当词汇库非常大的时候，相应的每个独热向量将会是巨大的一个维度
```

人们想到了使用共现矩阵降维分解来解决独热向量面临的这两个问题。

### 共现矩阵

首先需要明确以下几个概念概念：

```
1.窗口：共现矩阵看一句话的范围。
2.中心词：创造共现矩阵每一次会看将语料中的一个词作为中心词，而窗口之外的都是外围词。
3.外围词：每一次看的一个窗口中除了中心词之外都是外围词
4.插入开头节点和结尾节点
5.预料中的每一个词的位置都作为中心词看一次
6.需要规定分割，通常在word2vec训练里面一句话代表了一条数据
7.矩阵中的每一个元素代表的是一个词作为中心词（横坐标），对应的词作为外围词（纵坐标）的次数
```

下面举一个贡献矩阵的例子，假设有这样的一个预料：

```
I wanna fly in the sky.
The sky is blue.
I look at the sky through my sunglasses
```

假如我们指定相应的窗口是1那么对应的共现矩阵就应该是：

|            | I    | Wanna | fly  | in   | the  | sky  | is   | blue | look | at   | through | my   | Sunglasses |
| :--------: | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------- | ---- | ---------- |
|     I      | 0    | 1     | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0       | 0    | 0          |
|   wanna    | 1    | 0     | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0       | 0    | 0          |
|    fly     | 1    | 1     | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0       | 0    | 0          |
|     in     | 0    | 0     | 1    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0       | 0    | 0          |
|    the     | 0    | 0     | 0    | 1    | 0    | 2    | 0    | 0    | 0    | 1    | 0       | 0    | 0          |
|    sky     | 0    | 0     | 0    | 0    | 2    | 0    | 1    | 0    | 0    | 0    | 1       | 0    | 0          |
|     is     | 0    | 0     | 0    | 0    | 0    | 1    | 0    | 1    | 0    | 0    | 0       | 0    | 0          |
|    blue    | 0    | 0     | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0       | 0    | 0          |
|    look    | 1    | 0     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0       | 0    | 0          |
|     at     | 0    | 0     | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0    | 0       | 0    | 0          |
|  through   | 0    | 0     | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0       | 1    | 0          |
|     my     | 0    | 0     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1       | 0    | 1          |
| sunglasses | 0    | 0     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0       | 1    | 0          |

通过观察我们发现，这个共现矩阵里面有很多的零元素。

在矩阵降维的方法里面有一种叫做SVD的方法，使用SVD分解可以使矩阵的维度下降，并把其中有用的信息提取出来。

同时也可以使用PCA降维的方式。

### 通过CBOW与Skipgram训练词向量矩阵

其实CBOW与skipgram都是使用了与共现矩阵相似的提取词向量的方法。

他们之间在计算loss的过程其实就相差了一个平均的区别。

![alt text](https://github.com/OlgaChernytska/word2vec-pytorch/raw/main/docs/cbow_detailed.png)

![alt text](https://github.com/OlgaChernytska/word2vec-pytorch/raw/main/docs/skipgram_detailed.png)







# 代码实现整体思路

整体的代码的主要实现模块可以分为：数据集准备、训练、测试。

### 数据集准备

查看WikiText2的官方文档可以发现，里面的文本数据通常被分成了三份，分别是：

```python
train # 寻练使用数据集
valid # 验证使用数据集
test # 测试使用数据集（本模型验证不用）
```

其中需要分清楚的是

其中的测试集与验证集是不一样的。训练过程中使用的是验证集来进行训练，它的作用是进行快速调参。比如说选择一些合适的超参数(batchsize, lr等等)。但是一定不可以使用测试集做验证集，测试集一般是在模型训练结束之后才进行使用的。

对于此处创建的这个word2vec模型来说，测试的过程不需要用到测试集。因为词向量本身就是一个去表征语义的方法，而用与训练集具有同样数据的测试集再去进行验证，并不能证明我们训练出来的模型是否有效。我们使用的测试方法是用我们训练得到的embedding去做下游任务，比如文本的情感分类。

以torchtext.datasets.WikiText2为例，我们只需要使用其提供的接口就可以实现数据集的导入，也可以通过自行下载的方式来获得。

https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

```python
from torchtext.datasets import WikiText2
```

这里贴上官方文档中WikiText2的源码：

https://pytorch.org/text/stable/_modules/torchtext/datasets/wikitext2.html#WikiText2

```python
import os
from functools import partial
from typing import Union, Tuple

from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

if is_module_available("torchdata"):
    from torchdata.datapipes.iter import FileOpener, IterableWrapper
    from torchtext._download_hooks import HttpReader

URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"

MD5 = "542ccefacc6c27f945fb54453812b3cd"

NUM_LINES = {
    "train": 36718,
    "valid": 3760,
    "test": 4358,
}

DATASET_NAME = "WikiText2"

_EXTRACTED_FILES = {
    "train": os.path.join("wikitext-2", "wiki.train.tokens"),
    "test": os.path.join("wikitext-2", "wiki.test.tokens"),
    "valid": os.path.join("wikitext-2", "wiki.valid.tokens"),
}


def _filepath_fn(root, _=None):
    return os.path.join(root, os.path.basename(URL))


def _extracted_filepath_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, x):
    return _EXTRACTED_FILES[split] in x[0]


[docs]@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "valid", "test"))
def WikiText2(root: str, split: Union[Tuple[str], str]):
    """WikiText2 Dataset

    .. warning::

        using datapipes is still currently subject to a few caveats. if you wish
        to use this dataset with shuffling, multi-processing, or distributed
        learning, please see :ref:`this note <datapipes_warnings>` for further
        instructions.

    For additional details refer to https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

    Number of lines per split:
        - train: 36718
        - valid: 3760
        - test: 4358

    Args:
        # 此处指定相应的dataset文件保存路径
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        # 返回指定的集合
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `valid`, `test`)

    :returns: DataPipe that yields text from Wikipedia articles
    :rtype: str
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    url_dp = IterableWrapper([URL])
    # cache data on-disk
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_filepath_fn, root),
        hash_dict={_filepath_fn(root): MD5},
        hash_type="md5",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_extracted_filepath_fn, root, split))
    # Extract zip and filter the appropriate split file
    cache_decompressed_dp = (
        FileOpener(cache_decompressed_dp, mode="b").load_from_zip().filter(partial(_filter_fn, split))
    )
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_decompressed_dp, encoding="utf-8")
    return data_dp.readlines(strip_newline=False, return_path=False).shuffle().set_shuffle(False).sharding_filter()
```

通过观察源代码可以发现，使用相应的方法指定文件的路径就可以把数据集提取出来，但是在测试这个模块的时候，我发现，提取得到的是一个我不认识的东西；

```
>>> print(type(iteror))
<class 'torch.utils.data.datapipes.iter.grouping.ShardingFilterIterDataPipe'>
```

查看官方文档也没有相应的教学。

但是通过阅读官方文档之后，查看使用torchdata.datapipes.iter.IterableWrapper的说明文档之后发现：

```
>>> print(IterableWrapper.__doc__)

    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

可以通过使用list函数将类似的对象类型取出，得到相应的一个数据集的列表。

```
>>> train_iteror = WikiText2("data/", split="train")
>>> valid_iteror = WikiText2("data/", split="valid")
>>> len(list(train_iteror))
36718
>>> len(list(valid_iteror))
3760
>>>
```

这里的train和valid的数据集规模就对应上了。

那么我们看看这里面的一个元素长什么样子吧：
首先不妨将相应的数据存入一个list变量中：

```
>>> train_list=list(train_iteror)
>>> valid_list=list(valid_iteror)
```

看一看其中的一个元素：

```
>>> train_list[0]
' \n'
>>> train_list[1]
' = Valkyria Chronicles III = \n'
>>> valid_list[0]
' \n'
>>> valid_list[1]
' = Homarus gammarus = \n'
>>>
```

可以发现这里面是一个个的字符串，代表一个个的句子，同时每个字符串后面有换行符。

根据观察进一步发现，在里面的句子会出现\<unk\>：

```
>>> valid_list[152]
' Starting in the 1960s and following the construction of highways that made <unk> easier , residents began to move away from downtown in favor of new housing subdivisions to the north . After strip commercial interests began to move downtown , the city worked to designate several areas as historic districts in the 1970s and 80s to preserve the architectural character of the city . The Meridian Historic <unk> and Landmarks Commission was created in 1979 , and the Meridian Main Street program was founded in 1985 . \n'
```

这个词就代表了所谓的低频词，在后续的处理过程中，我们也需要加入我们自己定义的词频来将数据集中的低频词转换成符号\<unk\>

```python
config.yaml设置：
|config.yaml # 存放各种初始化参数
  |--MIN_WORD_FREQUENCY # 指定进入词汇表中的词出现的最小频率
  ...
```

超过这个设置频率的词就要被变成符号\<unk\>

刚开始在测试使用提取相应的数据集的时候，发现

```python
torchtext.datasets.WikiText2(root: str= 'data', split: Union[Tuple[str], ste] = ('train', 'valid', 'test'))
```

获取得到数据集，我们就要将在数据集中出现过的词独立出来，每一个词用一个对应的Id来表示，构成一个字典。

在组成这个数据集的过程中，我们需要思考一个问题，我们应该构建几个字典？

首先我们知道需要建立映射关系的两个量分别是：

```python
str(word) : int(wordId) 
```

我们知道，在python中，字典其实就是相当于函数对应的映射关系，每个值的Id是独一无二，但是可能会有一样的值。所以由键得到值的查询速度(O(1))和由值寻找到对应的键的速度要快很多。所以由于我们要在这两个量之间经常性的转换，所以对于这两个量就需要构建两个字典以提升在运算时候的速度：

```python
Word2Ind = {}
Ind2Word = {}
```

还有一个时间优化上的小问题，虽然对于训练过程来说，所有语料的词在准备的过程中，我们已经将其放入字典内了。但是word2vec在测试的步骤中，比如SVM测试，就需要用到可能不在字典内的语料。这个时候我们就需要对测试的语料中的每一个词，到训练好的embedding的转换，这一个步骤的第一步就是，判断我们拿到的测试数据的每一个词在我们现有的词库（vocab）里面是否存在，而假如使用：

```python
if("word" not in vocab_dic):
     pass
```

这样每一个词，都需要判断一次，而通常测试的语料就至少有十万以上的词汇量，这样去遍历字典式的检索显然就是不可行的。

所以不妨定义一个函数：

```python
def word_is_in_vocab(word, vocab):
    if(vocab.get(word)==None):
        return False
    else:
        return True
```

知道如何构建字典，下一步就是我们需要将句子分割成一个个的词，这个不用自己去写，pytorch提供了相应的函数。

```
>>> import torchtext
>>> from torchtext.data import get_tokenizer
>>> valid_list[152]
' Starting in the 1960s and following the construction of highways that made <unk> easier , residents began to move away from downtown in favor of new housing subdivisions to the north . After strip commercial interests began to move downtown , the city worked to designate several areas as historic districts in the 1970s and 80s to preserve the architectural character of the city . The Meridian Historic <unk> and Landmarks Commission was created in 1979 , and the Meridian Main Street program was founded in 1985 . \n'
>>> tokenizer = get_tokenizer("basic_english")
>>> tokens = tokenizer(valid_list[152])
>>> tokens
['starting', 'in', 'the', '1960s', 'and', 'following', 'the', 'construction', 'of', 'highways', 'that', 'made', '<unk>', 'easier', ',', 'residents', 'began', 'to', 'move', 'away', 'from', 'downtown', 'in', 'favor', 'of', 'new', 'housing', 'subdivisions', 'to', 'the', 'north', '.', 'after', 'strip', 'commercial', 'interests', 'began', 'to', 'move', 'downtown', ',', 'the', 'city', 'worked', 'to', 'designate', 'several', 'areas', 'as', 'historic', 'districts', 'in', 'the', '1970s', 'and', '80s', 'to', 'preserve', 'the', 'architectural', 'character', 'of', 'the', 'city', '.', 'the', 'meridian', 'historic', '<unk>', 'and', 'landmarks', 'commission', 'was', 'created', 'in', '1979', ',', 'and', 'the', 'meridian', 'main', 'street', 'program', 'was', 'founded', 'in', '1985', '.']
>>>
```

一个句子就变成一个个词了。

那么将句子变成tokens的同时，我们还需要构建一个vocab字典，同时还需要将高频词剔除。

这里在pytorch的文档中，提供了一个一举两得的解决办法：

```python
from torchtext.vocab import build_vocab_from_iterator
def build_vocab(data_iter, tokenizer):
"""Builds vocabulary from iterator"""

    vocab = build_vocab_from_iterator(
            map(tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=MIN_WORD_FREQUENCY,)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

data_iter = get_data_iterator(ds_name, ds_type, data_dir)
tokenizer = get_english_tokenizer()

if not vocab:
    vocab = build_vocab(data_iter, tokenizer)
```

通过这样子操作我们就获得了相应的vocab对象，那么这个对象应该如何去使用呢？

查看官方文档，我们发现两个值得关注的方法：

```
>>> Ind2Word = vocab.get_itos()
>>> Word2Ind = vocab.get_stoi()
>>> len(Ind2Word)
28782
>>> Ind2Word[5]
'and'
>>> Word2Ind['and']
5
>>>
```

通过这两个函数我们就得到了我们需要的映射关系。

我们看到前面使用了

```
vocab.set_default_index()
```

这个方法，根据官方文档的解释，就是当遇到vocab 中不存在的词汇的时候，就会使用这个默认的键对应的值，所以就把它设为\<unk\>。

```
>>> vocab.set_default_index(vocab["<unk>"])
>>> vocab["121affawe"]
0
>>> vocab.lookup_token(0)
'<unk>'
```

可以发现，在设置完之后，当使用一个随便输入的，数据集中不存在的词的时候，获得的index自动就指向刚刚设定的默认值\<unk\>的位置

这样子我们在编写dataloader的时候思路就很清晰了



### mini-batch划分

我们知道，在训练的时候如果直接将全部参数放入，直接进行迭代训练将会得到非常不好的效果。最好的方法是使用minibatch来划分数据集。

在我们的模型的数据集中，mini-batch是对应mini-batch大小的一组句子，但是真正送入模型进行训练的，是句子中包含的所有窗口。我们看官方提供的mini-batch划分方法中，使用的是torch.utils.data.DataLoader这个模块。对应的用法是：

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

def addbatch(data_train,data_test,batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)#shuffle是是否打乱数据集，可自行设置

    return data_loader
# 设置batch
traindata=addbatch(traininput,trainlabel,1000) # 1000为一个batch_size大小为1000，训练集为10000时一个epoch会训练10次。
for epoch in range(EPOCHS):
    for step, data in enumerate(traindata):
        # 这里是训练过程
        pass
```

但是，在我们这个模型中，并不是一个句子对应一个lable的，首先我们需要把句子转换成一个index的列表(后面用到pytorch里面提供的embeddings就是使用的这个来生成对应的词向量)，然后在CBOW和Skipgram这两个模型中，我们选择窗口之后生成lable的方式是不相同的。而且有一个很严重的问题；

```
1.我们是使用Batch-size个句子的集合来作为一个batch？
1.还是使用很多个窗口的集合作为batch？
```

我们需要想一下，更新参数的步骤是每一个batch更新一次参数。而我们更新参数是为了得到更低的窗口内的loss？还是为了得到句子内最小的loss？很明显是后者！

所以假如我们用窗口集合作为batch，由于batch数量是固定的，但是每一个句子的长度是不固定的，就很有可能使得一个集合内句子被打断。

但是使用句子作为batch却又不能使用上述的划分方法了，因为我们一个句子里面需要转换多个信息。

以cbow为例，假设batch大小是3，窗口为1，我们就应该这样划分：

```
# 语料：
[['I', 'want', 'to', 'learn', 'NLP'],
['NLP', 'is', 'amazing'],
['This', 'is', 'a', 'corpus']
...
]
# 语料->indexs
[[0, 1, 2, 3, 4],
[4, 5, 6],
[7, 5, 8, 9]
...
]
# indexs->batchs
[[0, 1, 2, 3, 4],
[4, 5, 6],
[7, 5, 8, 9]]
...
# for a single batch->(CBOW or Skipgram) data and label (e.g. CBOW)
train_data:
[
[0, 2],
[1, 3],
[2, 4],
[4, 6],
[7, 8],
[5, 9]
]
train_label:
[1, 2, 3, 5, 5, 8]
```

理解这两个模型如何取样就理解这个label是怎么来的。

我们发现这里面的句子数量虽然可以固定取，但是到了取出对应模型的样本的时候，这个最终的"a batch train data"的长度就完全取决于窗口长度了。

所以我决定自己写batch划分的函数。

### 模型的构建

构建模型第一步需要考虑的就是模型的输入和输出，我们这个模型的输入应该是一个batch的数据，然后输出的是一个loss数字。

概括起来的步骤就是

```
1.word_indexs --> train_data_embeddings
2.train_data_embeddings --> posibilities
3.(posibilities, labels) --> loss
4.optimizer
```



假设CBow模型(win2)中有一个batch，那么它所对应的正向传播的过程应该是像下面这样子的：

```
a_batch_of_cbow:
train_data(3,4):
[[1, 2, 4, 5],
[2, 3, 5, 6],
[3, 4, 6, 7]]
labels(3):
[3, 4, 5]

1.word_indexs --> embeddings
# 经过embeddings层，把坐标转换成embeddings,这里假设embeddings_dim=3,也就是把每一个train_data中的元素转换成了一个embeddings_dim向量。
train_data(3,4,3):
[[[1.2, 0.3, 0.6], [1.5, 6.4, 9.5], [5.2, 1.3, 6.2], [3.3, 0.2, 0.4]],
 [[1.5, 6.4, 9.5], [2.0, 8.3, 0.4], [3.3, 0.2, 0.4], [0.4, 0.7, 6.2]],
 [[2.0, 8.3, 0.4], [5.2, 1.3, 6.2], [0.4, 0.7, 6.2], [0.1, 0.5, 9.2]]]
labels(3)
[3,4,5]

2.train_data_embeddings --> posibilities
# 在这个例子中的每四个词向量组合需要去进行某种运算得到他们的预测词向量的在整个vocab中的posibility,同时labels需要转换成相应的onehot向量来作为预测概率分布的标签。
y_hat(3, vocab_size):
[[0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2],
 [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
 [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]]
one_hot_labels(3, vocab_size):
[[0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0]]

3.(posibilities, labels) --> loss
loss = ?

4.optimizer
```

##### 1.word_indexs --> train_data_embeddings

首先我们来完成第一步，发现在pytorch中提供了相应的库torch.nn.Embedding来完成这一个操作。

```
This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
```

这不是正好符合我们的需求吗？

```
Input: (*)(∗), IntTensor or LongTensor of arbitrary shape containing the indices to extract

Output: (*, H)(∗,H), where * is the input shape and H=\text{embedding\_dim}H=embedding_dim
```

来看一下官方文档中使用Embedding的例子：

```
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
```

##### 2.train_data_embeddings --> posibilities

首先想到的是，在CBOW模型下，需要的是用(2*window_size)个向量来去得到一个概率分布，那么我们就先得提取这几个向量的特征，第一个想到的方式就是用平均值来提取：

```
>>> wordvecs
tensor([[ 0.9775, -1.1697,  0.4181],
        [-0.6667, -0.5605,  1.6711]], grad_fn=<EmbeddingBackward0>)
>>> wordvecs.mean(axis=1)
tensor([0.0753, 0.1480], grad_fn=<MeanBackward1>)
```

 最终需要把每个提取得到特征的向量转换成概率分布，我们就可以想到使用torch.nn.Linear来解决



# 参数选择

在进行测试的过程中，发现测试得到的精确度与window_size高度相关。这是由于窗口的选择让模型可以“看到”句子的语义。

但是由于在CBOW与Skipgram模型选取外围词，不论与中心词相应的距离是多少，都是平等的，也就是说，如果窗口过小，可能无法提取有效的语义特征；而如果窗口过大，则会导致模型提取了无关的语义特征。

所以这个窗口的选择是适中最好。

根据官网的选择：

```
Performance
The training speed can be significantly improved by using parallel training on multiple-CPU machine (use the switch '-threads N'). The hyper-parameter choice is crucial for performance (both speed and accuracy), however varies for different applications. The main choices to make are:

architecture: skip-gram (slower, better for infrequent words) vs CBOW (fast)
skip-gram: 训练速度更慢，对于出现频率小的词效果更好
cbow: 训练速度更快
the training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors)
softmax：对低频词效果更好
负采样：对维度更低的词向量效果更好
sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 1e-3 to 1e-5)
dimensionality of the word vectors: usually more is better, but not always
对于更大量的数据，使用高频词采样能够有更高的精确度。
context (window) size: for skip-gram usually around 10, for CBOW around 5
```

可以看到，两个模型分别选择最好效果的窗口是：

```
CBOW: 5
skip-gram: 10
```

其实联系到两个模型相对应的原理就容易想到，由于两个模型相对应的外围词的权重是一样的，那么在采样的过程中，距离过远的词本身对语义上没有对应的贡献，而在模型中使用的权重却与距离无关，所以导致模型可能识别把太远的词识别成了是相近意思的词。窗口过小则会导致一些本该进入词向量中的重要信息没有经过训练，本来距离相近的词会被模型判定为独立的。

以下是使用模型测试的结果：





### trainer.py

