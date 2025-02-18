import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import time
from torch.utils.data import Dataset, DataLoader
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

class SkipGramDataset(Dataset):
    """
    Skip-Gram数据集类。
    """
    def __init__(self, data, word_to_idx):
        self.data = data
        # 单词到其对应索引的字典
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 通过索引获取目标词和上下文词
        target, context = self.data[index]
        # 将目标词和上下文词转换为对应的索引
        target_idx = self.word_to_idx[target]
        context_idx = self.word_to_idx[context]
        # 返回目标词和上下文词的索引
        return target_idx, context_idx
    

class SkipGramModel(nn.Module):
    """
    Skip-Gram模型类。
    
    该类继承自PyTorch的nn.Module，用于实现Word2Vec的Skip-Gram模型。
    
    参数:
    - vocab_size: 词汇表大小
    - embedding_dim: 词嵌入的维度
    """
    def __init__(self, vocab_size, embedding_dim, device='cpu'):
        super(SkipGramModel, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 初始化词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        # 初始化线性变换层
        self.linear = nn.Linear(embedding_dim, vocab_size).to(device)

    def forward(self, inputs):
        """
        定义模型的前向传播过程。
        
        参数:
        - inputs: 输入的词的索引
        
        返回值:
        - log_probs: 输入词周围的上下文词的对数概率
        """
        inputs = inputs.to(self.device)
        # 通过嵌入层将词的索引转换为词向量
        embeds = self.embeddings(inputs)
        # 将词向量传递到线性层，得到输出
        out = self.linear(embeds)
        # 对输出应用log_softmax函数，得到上下文词的对数概率
        log_probs = nn.functional.log_softmax(out, dim=1)
        return log_probs

def prepare_data(sentences, window_size=3):
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
            # 上下文词范围
            context = sentence[max(0, i - window_size):i] + sentence[i+1:i+window_size+1]
            target = word
            # 将目标词和上下文词配对加入训练数据
            for w in context:
                data.append((target, w))
                
    return vocab, word_to_idx, idx_to_word, data

def train(model, dataloader, num_epochs=100, learning_rate=0.01):
    """
    模型训练函数。
    
    参数:
    - model: Skip-Gram模型
    - dataloader: 训练数据加载器，包含目标词和上下文词的配对
    - num_epochs: 训练轮数，默认为100轮
    - learning_rate: 学习率，默认为0.01
    
    返回值:
    - 训练完成的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到 GPU

    # 定义交叉熵损失函数，并将其移动到选定的设备上
    criterion = nn.CrossEntropyLoss().to(device)
    # 定义优化器，用于更新模型参数
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            target, context = batch
            target = target.to(device)  # 将目标数据移动到 GPU
            context = context.to(device)  # 将输入数据移动到 GPU
            # 清零模型的梯度
            model.zero_grad()
            # 前向传播，获取目标词的对数概率
            log_probs = model(target)
            # 计算损失
            loss = criterion(log_probs, context)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss},{int(time.time())}")
    # 保存模型
    torch.save(model.state_dict(), 'skip_gram_model.pth')
    
    return model

def train_skip_gram(file_path, output_file, embedding_dim=100, window_size=3, num_epochs=100, learning_rate=0.01,batch_size=64):
    """
    使用Skip-Gram模型训练词向量的函数。
    
    参数:
    - file_path: 分词文件路径
    - output_file: 输出文件路径
    - embedding_dim: 词嵌入的维度
    - window_size: 窗口大小
    - num_epochs: 训练轮数
    - learning_rate: 学习率
    - batch_size: 批次大小
    
    返回值:
    - word_vectors: 包含词汇表中每个词的嵌入向量的字典
    """
    # 读取分词文件
    tokenized_text = train_utils.read_tokenize_file(file_path)
    print('读取分词文件完成', len(tokenized_text), int(time.time()))
    vocab, word_to_idx, idx_to_word, data = prepare_data([tokenized_text], window_size)
    print('预处理完成', int(time.time()))

    dataset = SkipGramDataset(data, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkipGramModel(len(vocab), embedding_dim, device)
    print('初始化模型完成', int(time.time()))
    ## 加载已训练模型
    # model
    model.load_state_dict(torch.load('skip_gram_model.pth'))
    trained_model = train(model, dataloader, num_epochs, learning_rate)
    print('训练模型完成',int(time.time()))

    tensor = trained_model.embeddings.weight.data
    cpu_tensor = tensor.cpu()
    embeddings = cpu_tensor.tolist()
    train_utils.save_embeddings(embeddings, idx_to_word, output_file)



# 示例用法
if __name__ == "__main__":
    # 文件路径
    file_path = 'G:\\nlp\\zhwiki_simplified_seg_jieba_stopwords_1_1.txt'
    output_file = 'G:\\nlp\\seg_jieba_1_1_skip_embeddings_3.txt'
    start = int(time.time())
    print("start time",start)
    # 训练 skip-gram 模型并获取词向量
    train_skip_gram(file_path, embedding_dim=10, num_epochs=20, learning_rate = 0.01, output_file = output_file)
    end = int(time.time())
    print("end time", end)
    print("耗时", end - start )
    # print("Word Embeddings:")
    # print(embeddings)
