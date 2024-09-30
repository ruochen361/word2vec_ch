# utils.py
import numpy as np

def read_tokenize_file(file_path, chunk_size=10000):
    """
    读取并分词文件内容。
    
    :param file_path: 文件路径
    :param chunk_size: 每次读取的行数
    :return: 分词后的文本列表
    """
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



def load_word_vectors(file_path):
    """
    从文件中加载词向量

    :param file_path: 词向量文件路径
    :return: 包含词和向量的字典
    """
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors