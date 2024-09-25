import torch
from ltp import LTP
import time
import codecs

# 默认 huggingface 下载，可能需要代理

ltp = LTP("G:\\nlp\\base1")  # 默认加载 Small 模型
                        # 也可以传入模型的路径，ltp = LTP("/path/to/your/model")
                        # /path/to/your/model 应当存在 config.json 和其他模型文件

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

# # 自定义词表
# ltp.add_word("汤姆去", freq=2)
# ltp.add_words(["外套", "外衣"], freq=2)

# #  分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
# output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
# # 使用字典格式作为返回结果
# print(output.cws)  # print(output[0]) / print(output['cws']) # 也可以使用下标访问
# print(output.pos)
# print(output.sdp)

# 停用词
stopwords = []
with codecs.open("G:\\nlp\\stopwords_hit.txt", "r", "utf-8") as f:
    for line in f:
        stopwords.append(line.strip())


print("==========current time: ",int(time.time()))

target = codecs.open("G:\\nlp\\zhwiki_simplified_6_seg.txt", "w", "utf-8")
i = 0
with codecs.open("G:\\nlp\\zhwiki_simplified_6.txt", "r", "utf-8") as f:
    lines = f.readlines()
    for line in lines:
        i = i+1
        # 分词
        words = ltp.pipeline(line, tasks=["cws"])
        line = []
        for word in words.cws:
            if word not in stopwords:
                line.append(word)
                
        target.writelines(" ".join(line))
        
        # # 获取分词结果
        # segmented_words = words.cws
        # print(segmented_words)
        # # 将分词结果添加到词表中
        # target.writelines(" ".join(segmented_words))
    if i%10000 == 0:
        print(u"已分词写入%s 行"%i)

target.close()
print("==========current time: ",int(time.time()))


