import numpy as np
from gensim.models import KeyedVectors

# 加载预训练的词向量模型
model_path = 'CBOW_model.pth'  # 请替换为实际路径
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

def cosine_similarity(vec1, vec2):
    """ 计算两个向量之间的余弦相似度 """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def find_similar_words(target_word, topn=5):
    """ 输入一个词，返回最相近的词及词向量 """
    if target_word not in word_vectors:
        print(f"词 '{target_word}' 不在词向量模型中")
        return []
    
    target_vector = word_vectors[target_word]
    similarities = [(word, cosine_similarity(target_vector, word_vectors[word])) 
                    for word in word_vectors.vocab.keys()]
    # 排序并取前N个最相似的词
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[:topn]


if __name__ == "__main__":
    word_list = ['苹果', '老虎', '龙', '朋友', '舅爷', '秦朝', '四羊方尊', '李清照', '功唐不捐', '道', '火锅', '西湖醋鱼']

    for word in word_list:
        similar_words = find_similar_words(word)
        for w, similarity in similar_words:
            print(f"原词：{word}, 近义词: {w}, 相似度: {similarity}")