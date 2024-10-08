# 词向量实验 word2vec
1. 使用维基百科中文语料
2. 使用哈工大LTP，结巴分词进行分词
3. 通过word2vec训练词向量
   CBOW模型
   Skip-gram模型
4. 通过余弦相似度，二维可视化，类比实验评估词向量


```
word2vec_ch
├─ assess 
│  ├─ analogical.py 类比实验
│  ├─ similarity.py 余弦相似度
│  └─ visualize.py 二维可视化
├─ check_gpu.py
├─ corpus
│  ├─ jieba_segm.py 结巴分词
│  ├─ splitfile.py  分割文件
│  ├─ wordsegm.py ltp分词
│  └─ zhwikidata.py 维基百科中文预料处理
├─ huffman_tree.py
├─ model_train
│  ├─ CBOW.py   
│  ├─ CBOW_GPU.py  GPU版本
│  ├─ CBOW_model.pth
│  ├─ CBOW_model_10.pth
│  ├─ CBOW_model_20.pth
│  ├─ CBOW_model_20_gpu.pth
│  ├─ CBOW_optimize.py 
│  ├─ skip_gram.py 
│  ├─ skip_gram_gpu.py GPU版本
│  └─ skip_gram_model.pth
├─ README.md
└─ train_utils.py

```