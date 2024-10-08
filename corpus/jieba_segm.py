import jieba
import jieba.posseg as pseg
import jieba.analyse
import codecs
import time


# 停用词
stopwords = []
with codecs.open("G:\\nlp\\stopwords_hit.txt", "r", "utf-8") as f:
    for line in f:
        stopwords.append(line.strip())

## 结巴分词

if __name__ == '__main__':
    f = codecs.open('G:\\nlp\\zhwiki_simplified.txt','r','utf-8')
    target = codecs.open('G:\\nlp\\zhwiki_simplified_seg_jieba_stopwords.txt','w','utf-8')
    print("start time",int(time.time()))
    linenum = 1
    line = f.readline()
    while line:
        line = line.strip()
        if len(line) > 0:
            linenum += 1
            seg_list = jieba.cut(line, cut_all=False)
            seg = []
            for word in seg_list:
                if word not in stopwords:  # 停用词过滤
                    seg.append(word)
            seg_line = " ".join(seg)
            seg_line = seg_line + "\n"
            target.writelines(seg_line)
        if linenum % 10000 == 0:
            print(linenum)
        line = f.readline()

    print("end time",int(time.time()))
    f.close()
    target.close()