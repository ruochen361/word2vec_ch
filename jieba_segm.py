import jieba
import jieba.posseg as pseg
import jieba.analyse
import codecs
import time

## 结巴分词

# 停用词
stopwords = []
with codecs.open("E:\postgradute\\nlp\\stopwords_hit.txt", "r", "utf-8") as f:
    for line in f:
        stopwords.append(line.strip())

if __name__ == '__main__':
    f = codecs.open('E:\postgradute\\nlp\\zhwiki_simplified.txt','r','utf-8')
    target = codecs.open('E:\postgradute\\nlp\\zhwiki_simplified_seg_jieba_stopword.txt','w','utf-8')
    print("start time",int(time.time()))
    linenum = 0
    line = f.readline()
    while line:
        line = line.strip()
        if len(line) > 0:
            linenum += 1
            seg_list = jieba.cut(line, cut_all=False)
            # seg = " ".join(seg_list)
            # target.writelines(seg)
            seg_line = ''
            for word in seg_list: 
                if word not in stopwords:
                    seg_line += word + " "
            target.writelines(seg_line)
            if linenum % 1000000 == 0:
                print(linenum)
        line = f.readline()

    print("end time",int(time.time()))
    f.close()
    target.close()