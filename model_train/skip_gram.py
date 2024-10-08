import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import random

class SkipGramModel(nn.Module):
    """
    Skip-Gram模型类。
    
    该类继承自PyTorch的nn.Module，用于实现Word2Vec的Skip-Gram模型。
    
    参数:
    - vocab_size: 词汇表大小
    - embedding_dim: 词嵌入的维度
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 初始化词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 初始化线性变换层
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        """
        定义模型的前向传播过程。
        
        参数:
        - inputs: 输入的词的索引
        
        返回值:
        - log_probs: 输入词周围的上下文词的对数概率
        """
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = nn.functional.log_softmax(out, dim=1)
        return log_probs

def prepare_data(sentences, window_size=2):
    """
    数据预处理函数。
    
    参数:
    - sentences: 由句子组成的列表，每个句子是由词组成的列表
    - window_size: 窗口大小，决定上下文词的数量
    
    返回值:
    - vocab: 词汇表列表
    - word_to_idx: 词到索引的映射字典
    - idx_to_word: 索引到词的映射字典
    - data: 训练数据，包含目标词和上下文词的配对
    """
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    # 构建词汇表
    vocab = list(word_counts.keys())
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    # 生成训练数据
    data = []
    for sentence in sentences:
        for i, word in enumerate(sentence):
            context = sentence[max(0, i - window_size):i] + sentence[i+1:i+window_size+1]
            target = word
            for w in context:
                data.append((target, w))
                
    return vocab, word_to_idx, idx_to_word, data

def train(model, data, word_to_idx, epochs=100, learning_rate=0.01):
    """
    模型训练函数。
    
    参数:
    - model: Skip-Gram模型
    - data: 训练数据，包含目标词和上下文词的配对
    - word_to_idx: 词到索引的映射字典
    - epochs: 训练轮数
    - learning_rate: 学习率
    
    返回值:
    - 训练完成的模型
    """
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for target, context in data:
            model.zero_grad()
            
            target = torch.tensor([word_to_idx[target]], dtype=torch.long)
            context = torch.tensor([word_to_idx[context]], dtype=torch.long)
            
            log_probs = model(target)
            loss = criterion(log_probs, context)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")
    
    # 保存模型
    torch.save(model.state_dict(), 'skip_gram_model.pth')
    
    return model

def train_skip_gram(sentences, embedding_dim=100, window_size=2, epochs=100, learning_rate=0.01):
    """
    使用Skip-Gram模型训练词向量的函数。
    
    参数:
    - sentences: 由句子组成的列表，每个句子是由词组成的列表
    - embedding_dim: 词嵌入的维度
    - window_size: 窗口大小
    - epochs: 训练轮数
    - learning_rate: 学习率
    
    返回值:
    - word_vectors: 包含词汇表中每个词的嵌入向量的字典
    """
    vocab, word_to_idx, idx_to_word, data = prepare_data([sentences], window_size)
    model = SkipGramModel(len(vocab), embedding_dim)

    ## 加载已训练模型
    # model.load_state_dict(torch.load('skip_gram_model.pth'))
    trained_model = train(model, data, word_to_idx, epochs, learning_rate)
    embeddings = trained_model.embeddings.weight.data
    return {word: embeddings[idx].tolist() for word, idx in word_to_idx.items()}


# 示例用法
if __name__ == "__main__":
    # 示例文本
    text = "包含 词汇表 中 每 个 词 的 嵌入 向量 的 字典"
    tokenized_text = text.split()

    word_vectors = train_skip_gram(tokenized_text, embedding_dim=50, epochs=100)
    print(word_vectors)