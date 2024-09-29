import numpy as np

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

def analogical_reasoning(word_vectors, a, b, c, topn=5):
    """
    执行词向量类比运算 a - b + c
    
    :param word_vectors: 词向量字典
    :param a: 词 a
    :param b: 词 b
    :param c: 词 c
    :param topn: 返回最相似的前 N 个词
    :return: 最相似的词列表
    """
    if a not in word_vectors or b not in word_vectors or c not in word_vectors:
        raise ValueError(f"词 {a}, {b} 或 {c} 不在词向量字典中")

    # 计算 a - b + c 的向量
    target_vector = word_vectors[a] - word_vectors[b] + word_vectors[c]

    # 计算所有词与目标向量的余弦相似度
    similarities = [(word, cosine_similarity(target_vector, vector))
                    for word, vector in word_vectors.items()]

    # 排序并取前 N 个最相似的词
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[:topn]

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)



if __name__ == "__main__":
    # 示例数据
    file_path = 'E:\postgradute\\nlp\\seg_jieba_1_1_embeddings_2.txt'  # 替换为实际文件路径
    word_vectors = load_word_vectors(file_path)

    # 执行类比实验
    list = [        
        ['皇帝', '男人', '女人'],
        ['大鱼', '小鱼', '小鱼'],
        ['猫', '猫科动物', '狗'],
        ['作家', '小说', '歌手'],
        ['秦朝', '郡县制', '汉朝'],
        ['儒家', '论语', '杂家'],
    ]
    for group in list:
        try:
            similar_words = analogical_reasoning(word_vectors, group[0], group[1], group[2], 3)
            print(f"词向量类比实验: {group[0]} - {group[1]} + {group[2]}")
            for word, similarity in similar_words:
                print(f"词: {word}, 相似度: {similarity:.4f}")
        except ValueError as e:
            print(e)