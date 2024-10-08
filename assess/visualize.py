import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import train_utils

# 配置 Matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def visualize_words(words, vectors):
    """
    对给定的词列表和对应的词向量进行二维可视化
    
    :param words: 词列表
    :param vectors: 对应的词向量列表
    """
    # 确保词向量和词列表长度一致
    assert len(words) == len(vectors), "词列表和词向量列表长度不一致"

    # 使用t-SNE降维到二维
    perplexity = min(30, len(vectors) - 1)
    tsne = TSNE(perplexity=perplexity, n_components=2, random_state=0)
    vectors = np.array(vectors)
    vectors_2d = tsne.fit_transform(vectors)

    # 绘制词向量
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    plt.title("2D Visualization of Word Vectors")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # 示例数据
    # words = ['李白', '杜甫', '白居易', '水稻', '小麦', '高粱', '张', '诸葛','柴', '盐', '茶','瞒天过海','树上开花',"围魏救赵"]
    words = ['李白', '杜甫', '白居易', '水稻', '小麦', '高粱', '张', '诸葛','柴', '盐', '茶', ]

    file_path = 'E:\postgradute\\nlp\\seg_jieba_1_1_embeddings_4.txt'  # 替换为实际文件路径
    word_vectors = train_utils.load_word_vectors(file_path)

    # vectors = [word_vectors[word] for word in words]
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    # 调用可视化函数
    visualize_words(words, vectors)