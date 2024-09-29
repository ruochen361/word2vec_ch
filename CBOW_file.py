import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn.functional as F
import time 

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).mean(dim=1)
        out = self.linear(embeds)
        return out

class CBOWDataset(Dataset):
    def __init__(self, tokenized_text, word_to_idx, window_size=2):
        self.tokenized_text = tokenized_text
        self.window_size = window_size
        self.word_to_idx = word_to_idx
        self.data = []
        self._build_data()

    def _build_data(self):
        indexed_words = [self.word_to_idx.get(w, 0) for w in self.tokenized_text]
        for i, target_word in enumerate(indexed_words):
            context_words = indexed_words[max(0, i - self.window_size):i] + indexed_words[i+1:min(len(indexed_words), i + self.window_size + 1)]
            if len(context_words) > 0:
                self.data.append((context_words, target_word))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]
        return context_words, target_word

def pad_sequence(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_sequences), [len(seq) for seq in sequences]

def collate_fn(batch):
    contexts, targets = zip(*batch)
    padded_contexts, _ = pad_sequence(contexts)
    targets = torch.tensor(targets)
    return padded_contexts, targets

def train(model, dataloader, num_epochs=100, learning_rate=0.001):
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
    torch.save(model.state_dict(), 'CBOW_model.pth')
    return model

def read_tokenize_file(file_path, chunk_size=10000):

    tokenized_text = []
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            lines = file.readlines(chunk_size)
            if not lines:
                break
            for line in lines:
                words = line.strip().split()
                tokenized_text.extend(words)
    return tokenized_text

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


def save_embeddings(embeddings, idx_to_word, output_file):
    """
    将词向量保存到文件。
    
    :param embeddings: 词向量矩阵
    :param idx_to_word: 词索引到词的映射
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, word in idx_to_word.items():
            vector = embeddings[idx]
            f.write(f"{word} {' '.join(map(str, vector))}\n")


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
    tokenized_text = read_tokenize_file(file_path, chunk_size)
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
    model = CBOWModel(vocab_size, embedding_dim)
    print('初始化模型完成',int(time.time()))
    # 训练模型
    trained_model = train(model, dataloader, num_epochs, learning_rate)
    print('训练模型完成',int(time.time()))
    # 训练后的词向量
    embeddings = trained_model.embedding.weight.data.numpy()

    save_embeddings(embeddings, idx_to_word, output_file)


# 示例用法
if __name__ == "__main__":
    # 文件路径
    file_path = 'E:\postgradute\\nlp\\zhwiki_simplified_6_seg_ltp_1.txt'
    output_file = 'E:\postgradute\\nlp\\embeddings.txt'
    print("start time",int(time.time()))
    # 训练 CBOW 模型并获取词向量
    train_cbow(file_path, embedding_dim=10, num_epochs=3, learning_rate = 0.1,output_file=output_file)
    print("end time",int(time.time()))
    # print("Word Embeddings:")
    # print(embeddings)