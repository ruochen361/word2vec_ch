import jieba
import jieba.posseg as pseg
import jieba.analyse
import codecs
import time

## 结巴分词

if __name__ == '__main__':
    f = codecs.open('G:\\nlp\\zhwiki_simplified_6.txt','r','utf-8')
    target = codecs.open('G:\\nlp\\zhwiki_simplified_6_seg.txt','w','utf-8')
    print("start time",int(time.time()))
    linenum = 1
    line = f.readline()
    while line:
        if len(line) > 0:
            linenum += 1
            seg_list = jieba.cut(line, cut_all=False)
            seg = " ".join(seg_list)
            target.writelines(seg)
        if linenum % 10000 == 0:
            print(linenum)
        linenum = linenum+1
        line = f.readline()

    print("end time",int(time.time()))
    f.close()
    target.close()