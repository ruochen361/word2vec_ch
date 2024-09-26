import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        初始化 CBOW 模型。
        
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词向量维度
        """
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        """
        前向传播计算。
        
        :param inputs: 输入的上下文词的索引
        :return: 输出的概率分布
        """
        embeds = self.embedding(inputs).mean(dim=1)
        out = self.linear(embeds)
        return out

class CBOWDataset(Dataset):
    def __init__(self, tokenized_text, window_size=2):
        """
        初始化 CBOW 数据集。
        
        :param tokenized_text: 分词后的文本列表
        :param window_size: 上下文窗口大小
        """
        self.tokenized_text = tokenized_text
        self.window_size = window_size
        self.data = []

        # 构建词汇表
        vocab = list(set(tokenized_text))
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

        # 转换文本为索引
        indexed_words = [self.word_to_idx[w] for w in tokenized_text]

        # 生成 CBOW 输入输出对
        for i, target_word in enumerate(indexed_words):
            context_words = indexed_words[max(0, i - window_size):i] + indexed_words[i+1:min(len(indexed_words), i + window_size + 1)]
            if len(context_words) > 0:
                self.data.append((context_words, target_word))

    def __len__(self):
        """
        返回数据集中样本的数量。
        
        :return: 样本数量
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。
        
        :param idx: 索引
        :return: 上下文词和目标词的索引
        """
        context_words, target_word = self.data[idx]
        return context_words, target_word

def pad_sequence(sequences, padding_value=0):
    """
    对一批次的序列进行填充，使其长度相同。
    
    :param sequences: 一批次的序列
    :param padding_value: 填充值，默认为 0
    :return: 填充后的张量和对应的长度
    """
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    lengths = []

    for seq in sequences:
        length = len(seq)
        padded_seq = seq + [padding_value] * (max_len - length)
        padded_sequences.append(padded_seq)
        lengths.append(length)

    return torch.tensor(padded_sequences), lengths

def collate_fn(batch):
    """
    自定义的 collate 函数，用于处理不同长度的上下文词。
    
    :param batch: 一批次的数据
    :return: 处理后的上下文词和目标词
    """
    contexts, targets = zip(*batch)
    padded_contexts, _ = pad_sequence(contexts)
    targets = torch.tensor(targets)
    return padded_contexts, targets

def train(model, dataloader, num_epochs=100, learning_rate=0.001):
    """
    训练 CBOW 模型并返回词向量。
    
    :param model: 待训练的模型
    :param dataloader: 数据加载器
    :param num_epochs: 训练轮数，默认为 100
    :param learning_rate: 学习率，默认为 0.001
    :return: 训练后的模型
    """

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充值
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        # 遍历数据加载器中的每个样本，进行前向传播和反向传播
        for context, target in dataloader:
            optimizer.zero_grad()  # 清零梯度
            output = model(context)  # 前向传播
            loss = loss_function(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失值
        
        # 打印每个训练轮次的损失
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # 保存训练完成后的模型参数
    torch.save(model.state_dict(), 'CBOW_model.pth')
    return model

def train_cbow(tokenized_text, embedding_dim=100, window_size=2, batch_size=32, num_epochs = 100, learning_rate=0.001):
   
    """
    训练CBOW模型以获取词嵌入。

    参数:
    - tokenized_text: 分词后的文本，用于模型训练。
    - embedding_dim: 词嵌入的维度，默认为100。
    - window_size: 窗口大小，决定了上下文单词的数量，默认为2。
    - batch_size: 每个训练批次的样本数量，默认为32。
    - num_epochs: 训练的轮数，默认为100。
    - learning_rate: 学习率，控制模型训练的速度，默认为0.001。

    返回:
    - embeddings: 训练后的词嵌入矩阵。
    """
   
    # 创建数据集
    dataset = CBOWDataset(tokenized_text, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型
    vocab_size = len(dataset.word_to_idx)
    model = CBOWModel(vocab_size, embedding_dim)

    ## 加载已训练模型
    # model.load_state_dict(torch.load('skip_gram_model.pth'))
    trained_model = train(model, dataloader, num_epochs, learning_rate)
    embeddings = trained_model.embeddings.weight.data

    return embeddings


# 示例用法
if __name__ == "__main__":
    # 示例文本
    text = "包含 词汇表 中 每 个 词 的 嵌入 向量 的 字典"
    tokenized_text = text.split()

    # 训练 CBOW 模型并获取词向量
    embeddings = train_cbow(tokenized_text, embedding_dim=50, num_epochs=50)
    print("Word Embeddings:")
    print(embeddings)