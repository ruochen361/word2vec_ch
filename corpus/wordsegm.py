import torch
from ltp import LTP
import time
import codecs


## 哈工大 ltp 分词
ltp = LTP("G:\\nlp\\base1")  

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")


# 停用词
stopwords = []
with codecs.open("G:\\nlp\\stopwords_hit.txt", "r", "utf-8") as f:
    for line in f:
        stopwords.append(line.strip())


if __name__ == '__main__':
    f = codecs.open('G:\\nlp\\zhwiki_simplified_6.txt','r','utf-8')
    target = codecs.open('G:\\nlp\\zhwiki_simplified_6_seg_ltp.txt','w','utf-8')
    print("start time",int(time.time()))
    linenum = 1
    line = f.readline()
    while line:
        if len(line) > 0:
            linenum += 1
            words = ltp.pipeline(line, tasks=["cws"])
            seg = " ".join(words.cws)
            target.writelines(seg)

            if linenum % 10000 == 0:
                print(linenum)
        line = f.readline()

    print("end time",int(time.time()))
    f.close()
    target.close()