import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn.functional as F
import time 
import train_utils
import huffman_tree

class CBOWModel(nn.Module):

     """
    CBOW模型类，用于词向量训练
    
    该类通过CBOW(Continuous Bag-of-Words)模型来预测给定单词的上下文单词，包含词嵌入层和线性层，
    支持使用霍夫曼树路径进行负采样训练。
    
    参数:
    vocab_size (int): 词汇表大小，用于设置嵌入层的大小
    embedding_dim (int): 词嵌入的维度
    device (str, optional): 设备类型，如'cpu'或'cuda'，用于指定模型在哪个设备上运行，默认为'cpu'
    huffman_paths (dict, optional): 霍夫曼树的路径字典，用于负采样训练，默认为None
    negative_samples (int, optional): 负采样数量，默认为5
    """
     def __init__(self, vocab_size, embedding_dim, device='cpu', huffman_paths=None, negative_samples=5):
        """
        初始化CBOW模型，设置模型的基本参数和结构
        
        该构造函数初始化了模型运行的设备，词嵌入层，线性层，霍夫曼路径和负采样数量。
        """
        super(CBOWModel, self).__init__()
        self.device = device  # 设置模型运行的设备
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)  # 初始化词嵌入层，并移动到指定设备
        self.linear = nn.Linear(embedding_dim, 1).to(device)  # 初始化线性层，并移动到指定设备
        self.huffman_paths = huffman_paths  # 设置霍夫曼路径，用于负采样训练
        self.negative_samples = negative_samples  # 设置负采样数量

     def forward(self, inputs, target, negative_samples):
        """
        前向传播函数，用于计算输入样本与目标样本及负样本之间的相似度得分。
        
        参数:
        - inputs: 输入样本的索引，通常是一个批次的数据。
        - target: 目标样本的索引。
        - negative_samples: 负样本的索引列表。
        
        返回值:
        - positive_scores: 输入样本与目标样本之间的相似度得分。
        - negative_scores: 输入样本与每个负样本之间的相似度得分。
        """
        # 将输入样本转移到指定的设备（CPU或GPU）
        inputs = inputs.to(self.device)
        # 对输入样本进行嵌入，并在序列维度上取平均，得到输入样本的嵌入表示
        embeds = self.embedding(inputs).mean(dim=1)
        # 获取目标样本的嵌入表示
        target_embed = self.embedding(target)
        # 获取负样本的嵌入表示
        negative_embed = self.embedding(negative_samples)
        
        # 计算输入样本与目标样本之间的点积，得到相似度得分
        positive_scores = torch.sum(embeds * target_embed, dim=1)
        # 计算输入样本与每个负样本之间的点积，得到相似度得分，并进行适当的维度调整
        negative_scores = torch.sum(embeds.unsqueeze(1) * negative_embed, dim=2).squeeze()
        
        # 返回输入样本与目标样本及负样本之间的相似度得分
        return positive_scores, negative_scores

def hierarchical_softmax_loss(outputs, target, huffman_paths, vocab_size):
    """
    计算层次softmax损失函数。

    层次softmax是一种用于处理大规模分类问题的技术，它通过构建词汇的哈夫曼树，
    将对数几率损失函数分解为一系列的二元分类任务，以降低计算复杂度。

    参数:
    - outputs: 模型的输出，形状为(batch_size, max_path_length)。
    - target: 目标标签，形状为(batch_size,)。
    - huffman_paths: 词汇表中每个词的哈夫曼路径字典，用于确定通过哈夫曼树的路径。
    - vocab_size: 词汇表大小，用于某些特定计算，本函数中未直接使用。

    返回:
    - loss: 层次softmax损失值。
    """
    device = outputs.device  # 获取当前设备（CPU或GPU）
    loss = 0  # 初始化损失值
    # 遍历每个样本
    for i in range(outputs.size(0)):
        path = huffman_paths[target[i]]  # 获取当前样本目标词的哈夫曼路径
        # 遍历哈夫曼路径上的每个节点
        for j, bit in enumerate(path):
            idx = int(bit)  # 当前节点的标签（0或1）
            output = outputs[i, j]  # 当前样本在当前节点的输出
            # 根据层次softmax损失公式更新损失值
            loss += -F.logsigmoid(output if idx == 0 else -output)
    return loss.mean()  # 返回平均损失值

def combined_loss(positive_scores, negative_scores):
    """
    计算组合损失函数，该函数用于同时考虑正样本和负样本的损失，
    以优化模型参数。

    参数:
    positive_scores (Tensor): 正样本的分数。
    negative_scores (Tensor): 负样本的分数。

    返回:
    Tensor: 正样本损失和负样本损失的和。
    """
    # 计算正样本的损失，使用-logsigmoid函数
    positive_loss = -F.logsigmoid(positive_scores).mean()
    # 计算负样本的损失，使用-logsigmoid函数，注意符号的相反
    negative_loss = -F.logsigmoid(-negative_scores).mean()
    # 返回正样本损失和负样本损失的和
    return positive_loss + negative_loss


class CBOWDataset(Dataset):
    def __init__(self, tokenized_text, word_to_idx, window_size=2, huffman_paths=None, negative_samples=5):
        self.tokenized_text = tokenized_text
        self.window_size = window_size
        self.word_to_idx = word_to_idx
        self.huffman_paths = huffman_paths
        self.negative_samples = negative_samples
        self.data = []
        self._build_data()

def _build_data(self):
    """
    构建训练数据集。
    
    本方法首先将文本中的每个词转换为其对应的索引，然后计算每个词的频率和概率，
    并根据这些概率生成负样本。随后，遍历每个词，收集其上下文词，构成训练数据对。
    
    属性:
    - indexed_words: 列表，存储每个词的索引。
    - freqs: 词频统计对象，记录每个词出现的频率。
    - total_words: 整数，所有词的总数。
    - prob: 字典，存储每个词的概率，即频率除以总词数。
    - negatives: 字典，根据词的概率生成的负样本数量。
    - data: 列表，存储训练数据对，每个数据对由上下文词和目标词组成。
    """
    
    # 将文本中的每个词转换为对应的索引，如果词不在词典中，则使用0作为默认索引
    indexed_words = [self.word_to_idx.get(w, 0) for w in self.tokenized_text]
    
    # 统计每个词的频率
    freqs = Counter(indexed_words)
    
    # 计算所有词的总数
    total_words = sum(freqs.values())
    
    # 计算每个词的概率，即频率除以总词数
    prob = {word: freq / total_words for word, freq in freqs.items()}
    
    # 根据每个词的概率生成负样本数量，使用0.75次方对概率进行调整
    self.negatives = {word: int(prob[word] ** 0.75 * total_words) for word in freqs}
    
    # 遍历每个词，收集其上下文词
    for i, target_word in enumerate(indexed_words):
        # 确定上下文词的范围，避免越界
        context_words = indexed_words[max(0, i - self.window_size):i] + indexed_words[i+1:min(len(indexed_words), i + self.window_size + 1)]
        
        # 如果存在上下文词，则将上下文词和目标词构成的数据对添加到训练数据中
        if len(context_words) > 0:
            self.data.append((context_words, target_word))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]
        negative_samples = torch.multinomial(torch.tensor(list(self.negatives.values())), self.negative_samples, replacement=True)
        negative_samples = [list(self.negatives.keys())[i] for i in negative_samples]
        return context_words, target_word, negative_samples

def pad_sequence(sequences, padding_value=0):
    """
    对一组序列进行填充，使得所有序列的长度相同。
    
    参数:
    sequences (list of list): 需要被填充的序列列表。
    padding_value (int, optional): 用于填充的值，默认为0。
    
    返回:
    torch.tensor: 填充后的序列张量。
    list: 每个序列原始的长度列表。
    """
    # 计算最长序列的长度
    max_len = max(len(seq) for seq in sequences)
    
    # 对每个序列进行填充，使其长度等于最长序列的长度
    padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    
    # 返回填充后的序列张量和每个序列原始的长度列表
    return torch.tensor(padded_sequences), [len(seq) for seq in sequences]

def collate_fn(batch):
    """
    自定义的批处理函数，用于对数据集中的样本进行批处理，以便进行批量训练或评估。
    
    该函数接收一个批次的数据样本列表，每个数据样本是一个包含三个元素的元组：
    - contexts: 上下文信息
    - targets: 目标信息
    - negative_samples: 负样本信息
    
    函数的主要步骤包括：
    1. 解压批次数据，分别获取所有的上下文信息、目标信息和负样本信息。
    2. 对上下文信息进行填充（padding），以使得所有样本具有相同的序列长度。
    3. 将目标信息和负样本信息转换为PyTorch张量（Tensor）格式。
    
    参数:
    - batch: 一个列表，包含多个数据样本元组，每个元组包含一个上下文信息、一个目标信息和多个负样本信息。
    
    返回:
    - padded_contexts: 填充后的上下文信息序列。
    - targets: 目标信息的张量。
    - negative_samples: 负样本信息的张量列表。
    """
    # 解压批次数据，获取所有的上下文信息、目标信息和负样本信息
    contexts, targets, negative_samples = zip(*batch)
    
    # 对上下文信息进行填充，并获取填充后的序列
    padded_contexts, _ = pad_sequence(contexts)
    
    # 将目标信息转换为PyTorch张量格式
    targets = torch.tensor(targets)
    
    # 将每个负样本信息转换为PyTorch张量格式
    negative_samples = [torch.tensor(neg) for neg in negative_samples]
    
    # 返回填充后的上下文信息序列、目标信息张量和负样本信息张量列表
    return padded_contexts, targets, negative_samples

def train(model, dataloader, num_epochs=100, learning_rate=0.001):
    """
    训练模型。

    参数:
    model: 模型实例，要训练的模型。
    dataloader: 数据加载器，用于迭代加载训练数据。
    num_epochs: 整数，训练的轮数，默认为100。
    learning_rate: 浮点数，学习率，默认为0.001。

    返回:
    model: 训练完成后的模型实例。
    """
    # 根据CUDA设备的可用性选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型转移到选定的设备上
    model.to(device)
    # 初始化Adam优化器，学习率可由参数learning_rate设置
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 开始训练，迭代num_epochs次
    for epoch in range(num_epochs):
        total_loss = 0
        # 遍历数据加载器，获取训练数据
        for context, target, negative_samples in dataloader:
            # 将上下文、目标和负样本转移到选定的设备上
            context = context.to(device)
            target = target.to(device)
            negative_samples = negative_samples.to(device)

            # 零化梯度，为新一轮反向传播做准备
            optimizer.zero_grad()
            # 前向传播，获取正样本和负样本的分数
            positive_scores, negative_scores = model(context, target, negative_samples)
            # 计算损失
            loss = combined_loss(positive_scores, negative_scores)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累加损失，用于监控训练过程
            total_loss += loss.item()
        # 打印当前轮次和累计损失
        print(f"Epoch {epoch + 1}, Loss: {total_loss}, {int(time.time())}")
    # 训练完成后，保存模型参数
    torch.save(model.state_dict(), 'CBOW_model.pth')
    # 返回训练完成后的模型
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
    huffman_paths = huffman_tree.build_huffman_tree_and_paths(tokenized_text)
    dataset = CBOWDataset(tokenized_text, word_to_idx, window_size,huffman_paths=huffman_paths, negative_samples=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print('创建数据集完成',int(time.time()))
    # 初始化模型
    vocab_size = len(word_to_idx)
    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBOWModel(vocab_size, embedding_dim, device,huffman_paths=huffman_paths, negative_samples=5)
    # model.load_state_dict(torch.load('E:\workspace\\nlp\word2vec\CBOW_model_100.pth'))
    print('初始化模型完成',int(time.time()))
    # 训练模型
    trained_model = train(model, dataloader, num_epochs, learning_rate)
    print('训练模型完成',int(time.time()))
    # 训练后的词向量
    # trained_model = trained_model.to(device="cpu")
    tensor = trained_model.embedding.weight.data
    cpu_tensor = tensor.cpu()
    embeddings = cpu_tensor.tolist()
    # embeddings = trained_model.embedding.weight.data.numpy()

    train_utils.save_embeddings(embeddings, idx_to_word, output_file)


# 示例用法
if __name__ == "__main__":
    # 文件路径
    file_path = 'G:\\nlp\\zhwiki_simplified_seg_jieba_stopwords_1_1.txt'
    output_file = 'G:\\nlp\\seg_jieba_1_1_embeddings_4.txt'
    start = int(time.time())
    print("start time",start)
    # 训练 CBOW 模型并获取词向量
    train_cbow(file_path, embedding_dim=20, num_epochs=10, learning_rate = 0.01, output_file = output_file)
    end = int(time.time())
    print("end time", end)
    print("耗时", end - start )
    # print("Word Embeddings:")
    # print(embeddings)