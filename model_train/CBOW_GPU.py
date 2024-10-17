import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn.functional as F
import time 
import sys
from pathlib import Path

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()
# 获取当前文件所在的目录
current_dir = current_file_path.parent
# 获取项目根目录
root_dir = current_dir.parent
# 将项目根目录添加到系统路径中
sys.path.append(str(root_dir))

from assess import train_utils


class CBOWModel(nn.Module):
    """
    CBOW模型类，继承自PyTorch的nn.Module。

    该模型的目的是根据上下文单词来预测中心单词，利用词嵌入和线性变换实现。

    参数:
    - vocab_size: 词汇表大小，决定嵌入层和线性层的输入/输出维度。
    - embedding_dim: 词嵌入的维度，即每个单词将被表示为多少维的向量。
    - device: 执行计算的设备，默认为'cpu'，可根据需求设置为'cuda'以使用GPU。
    """

    def __init__(self, vocab_size, embedding_dim, device='cpu'):
        super(CBOWModel, self).__init__()
        self.device = device  # 记录设备信息，用于后续操作
        # 初始化嵌入层，将词汇表大小和词嵌入维度作为参数，指定设备进行初始化
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        # 初始化线性层，用于从嵌入向量空间转换到词汇表大小的输出，同样指定设备
        self.linear = nn.Linear(embedding_dim, vocab_size).to(device)

    def forward(self, inputs):
        """
        定义模型的前向传播过程。

        参数:
        - inputs: 输入的上下文单词的索引，形状为(batch_size, sequence_length)。

        返回:
        - out: 模型的输出，即预测的中心单词的概率分布，形状为(batch_size, vocab_size)。
        """
        inputs = inputs.to(self.device)  # 将输入转移到指定设备
        # 对输入的每个单词进行嵌入查找，并在序列维度上求平均，得到上下文的嵌入表示
        embeds = self.embedding(inputs).mean(dim=1)
        # 将上下文的嵌入表示通过线性层，得到预测的中心单词的概率分布
        out = self.linear(embeds)
        return out


class CBOWDataset(Dataset):
    """
    CBOW数据集类，用于将文本数据转换为CBOW模型所需的输入格式。

    CBOW模型需要上下文单词来预测中心单词，因此该数据集类的主要工作是构建这些上下文和目标对。

    参数:
    - tokenized_text: 已分词的文本，通常是一个字符串列表。
    - word_to_idx: 单词到索引的映射字典，用于将单词转换为整数。
    - window_size: 窗口大小，定义了上下文单词的范围，默认为2。
    """

    def __init__(self, tokenized_text, word_to_idx, window_size=2):
        self.tokenized_text = tokenized_text
        self.window_size = window_size
        self.word_to_idx = word_to_idx
        self.data = []
        self._build_data()

    def _build_data(self):
        """
        构建CBOW模型所需的数据集。

        该方法将文本中的单词转换为索引，并为每个中心单词创建一个上下文单词列表。
        如果上下文单词的数量大于0，则将其与目标单词一起添加到数据集中。
        """
        indexed_words = [self.word_to_idx.get(w, 0) for w in self.tokenized_text]
        for i, target_word in enumerate(indexed_words):
            context_words = indexed_words[max(0, i - self.window_size):i] + indexed_words[i + 1:min(len(indexed_words),
                                                                                                    i + self.window_size + 1)]
            if len(context_words) > 0:
                self.data.append((context_words, target_word))

    def __len__(self):
        """
        返回数据集中上下文和目标对的数量。

        返回值:
        - 数据集中样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引处的上下文和目标对。

        参数:
        - idx: 数据集中样本的索引。

        返回值:
        - 一个包含上下文单词列表和目标单词的元组。
        """
        context_words, target_word = self.data[idx]
        return context_words, target_word

def pad_sequence(sequences, padding_value=0):
    """
    对一组序列进行填充，使其长度一致。

    参数:
    sequences (list of list): 一个包含多个序列的列表，每个序列的元素可以是任何类型。
    padding_value (int, optional): 用于填充的值，默认为0。

    返回:
    torch.tensor: 填充后的序列张量，所有序列的长度都已扩展到最长序列的长度。
    list of int: 每个序列原始的长度列表。

    说明:
    - 该函数主要用于处理一批长度不一的序列数据，例如文本序列或时间序列数据。
    - 填充的目的是为了将这些序列转化为固定尺寸的数据，以便进行后续的批量处理。
    - 使用指定的padding_value对短序列进行填充，使得所有序列的长度都与最长的序列一致。
    """
    # 计算最长序列的长度
    max_len = max(len(seq) for seq in sequences)
    # 对每个序列进行填充，使其长度达到max_len
    padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    # 返回填充后的序列张量和每个序列原始的长度列表
    return torch.tensor(padded_sequences), [len(seq) for seq in sequences]


def collate_fn(batch):
    """
    自定义的批处理函数，用于在数据加载时处理每个批次的数据。

    该函数的主要作用是将一个批次的数据（由多个样本组成），
    转换为适用于神经网络训练的格式。具体来说，它会将每个样本的
    输入（context）和输出（target）分别提取出来，对输入进行填充操作
    以确保所有样本的输入长度一致，然后将输出转换为张量格式。

    参数:
    - batch: 一个批次的数据，由多个样本组成。每个样本是一个元组，
             其中包含一个输入（context）和一个输出（target）。

    返回:
    - padded_contexts: 填充后的输入序列，适用于神经网络的批量处理。
    - targets: 目标输出的张量，表示每个样本的标签或期望的输出。
    """
    # 将每个样本的输入（context）和输出（target）分别提取出来
    contexts, targets = zip(*batch)

    # 对输入序列进行填充操作，以确保所有输入的长度一致
    # 返回填充后的序列以及序列的实际长度（此处忽略了序列的实际长度）
    padded_contexts, _ = pad_sequence(contexts)

    # 将输出数据转换为张量格式
    targets = torch.tensor(targets)

    # 返回填充后的输入序列和目标输出的张量
    return padded_contexts, targets

def train(model, dataloader, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到GPU
    # 初始化损失函数，并忽略索引为0的输出（通常用于padding），然后将损失函数移动到选定的设备上
    loss_function = nn.CrossEntropyLoss(ignore_index=0).to(device)
    # 使用Adam优化器，传入模型参数和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in dataloader:
            context = context.to(device)  # 将输入数据移动到GPU
            target = target.to(device)  # 将目标数据移动到GPU
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播，得到模型输出
            output = model(context)
            # 计算损失
            loss = loss_function(output, target)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}, {int(time.time())}")
    torch.save(model.state_dict(), 'CBOW_model.pth')
    return model


def build_vocab(tokenized_text, min_freq=1):
    """
    构建词汇表。
    
    :param tokenized_text: 分词后的文本列表
    :param min_freq: 最小词频
    :return: 词汇表字典
    """
    counter = Counter(tokenized_text)
    vocab = [word for word, count in counter.items() if count >= min_freq]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word



def train_cbow(file_path, output_file, embedding_dim=100, window_size=2, batch_size=32, num_epochs=100, learning_rate=0.001, chunk_size=10000, min_freq=1):
    """
    训练CBOW模型以获取词嵌入。
    
    :param file_path: 文件路径
    :param embedding_dim: 词嵌入的维度，默认为100
    :param window_size: 窗口大小，默认为2
    :param batch_size: 每个训练批次的样本数量，默认为32
    :param num_epochs: 训练的轮数，默认为100
    :param learning_rate: 学习率，默认为0.001
    :param chunk_size: 每次读取的行数，默认为10000
    :param min_freq: 最小词频，默认为1
    :return: 训练后的词嵌入矩阵
    """
    # 读取分词文件
    tokenized_text = train_utils.read_tokenize_file(file_path, chunk_size)
    print('读取分词文件完成',len(tokenized_text),int(time.time()))
    # 构建词汇表
    word_to_idx, idx_to_word = build_vocab(tokenized_text, min_freq)
    print('构建词汇表完成',int(time.time()))
    # 创建数据集
    dataset = CBOWDataset(tokenized_text, word_to_idx, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print('创建数据集完成',int(time.time()))
    # 初始化模型
    vocab_size = len(word_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CBOWModel(vocab_size, embedding_dim,device)
    # model.load_state_dict(torch.load('D:\workspace\\nlp\word2vec_ch\CBOW_model.pth'))
    print('初始化模型完成',int(time.time()))
    # 训练模型
    trained_model = train(model, dataloader, num_epochs, learning_rate)
    print('训练模型完成',int(time.time()))
    # 训练后的词向量
    tensor = trained_model.embedding.weight.data
    cpu_tensor = tensor.cpu()
    embeddings = cpu_tensor.tolist()

    train_utils.save_embeddings(embeddings, idx_to_word, output_file)


# 示例用法
if __name__ == "__main__":
    # 文件路径
    file_path = 'G:\\nlp\\test_seg.txt'
    output_file = 'G:\\nlp\\test_embeddings.txt'
    start = int(time.time())
    print("start time",start)
    # 训练 CBOW 模型并获取词向量
    train_cbow(file_path, embedding_dim=1, num_epochs=1, learning_rate = 0.01, output_file = output_file)
    end = int(time.time())
    print("end time", end)
    print("耗时", end - start )
    # print("Word Embeddings:")
    # print(embeddings)