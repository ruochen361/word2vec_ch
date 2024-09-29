import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_words(words, vectors):
    """
    对给定的词列表和对应的词向量进行二维可视化
    
    :param words: 词列表
    :param vectors: 对应的词向量列表
    """
    # 确保词向量和词列表长度一致
    assert len(words) == len(vectors), "词列表和词向量列表长度不一致"

    # 使用t-SNE降维到二维
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    # 绘制词向量
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    plt.title("2D Visualization of Word Vectors")
    plt.show()

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



if __name__ == "__main__":
    # 示例数据
    # words = ['李白', '杜甫', '白居易', '水稻', '小麦', '高粱', '张', '诸葛','柴', '盐', '茶','瞒天过海','树上开花',"围魏救赵"]
    words = ['李白', '杜甫', '白居易', '水稻', '小麦', '高粱', '张', '诸葛','柴', '盐', '茶', ]

    file_path = 'G:\\nlp\\seg_jieba_1_1_embeddings.txt'  # 替换为实际文件路径
    word_vectors = load_word_vectors(file_path)

    # vectors = [word_vectors[word] for word in words]
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    # 调用可视化函数
    visualize_words(words, vectors)