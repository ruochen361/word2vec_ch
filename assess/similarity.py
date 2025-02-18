import numpy as np
from scipy.spatial.distance import cosine
import train_utils

def find_similar_words(word, word_vectors, top_n=10):
    """
    查找与给定词最相近的词。
    """
    word_vector = word_vectors.get(word)
    if word_vector is None:
        return []

    similarities = []
    for w, vec in word_vectors.items():
        if w != word:
            sim = 1 - cosine(word_vector, vec)
            similarities.append((w, sim, vec))

    # 排序并返回最相似的词
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def get_top_similar_words(words, word_vectors, top_n=10):
    """
    获取每个词最相近的词。
    """
    results = {}
    for word in words:
        similar_words = find_similar_words(word, word_vectors, top_n=top_n)
        results[word] = similar_words
    return results

if __name__ == "__main__":

    # 词向量文件路径
    file_path = 'G:\\nlp\\seg_jieba_1_1_embeddings_4.txt'

    word_list = ['苹果', '老虎', '龙', '朋友', '舅爷', '秦朝', '四羊方尊', '李清照', '功唐不捐', '道', '火锅', '西湖醋鱼']

    # 读取词向量
    word_vectors = train_utils.load_word_vectors(file_path)
    # 计算最相近的词
    top_similar_words = get_top_similar_words(word_list, word_vectors, top_n=3)

    # 输出结果
    for word, similar_words in top_similar_words.items():
        for sim_word, similarity,vector in similar_words:
            print(f"原词：{word}, 近义词: {sim_word}, 相似度: {similarity:.4f}，Vector: {vector}")