# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:38:04 2019

@author: 振振
"""
#导入所需要的库，nitk英文分词
import nltk
import sys
from nltk.corpus import brown
#对词库进行预处理，给词加上开始和结束符号
brown_tags_words = [ ]
for sent in brown.tagged_sents():
    brown_tags_words.append(("START","START"))
    brown_tags_words.extend([(tag[:2],word) for (word,tag) in sent ])
    brown_tags_words.append(("END","END"))
#使用nltk中的统计工具，把单词和其tag之间的关系进行统计
# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)#统计词频
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)#统计概率
#可以观察一下不同的词属于不同词性的概率
print("The probability of an adjective (JJ) being 'beautiful' is", cpd_tagwords["JJ"].prob("beautiful"))
print("The probability of a verb (VB) being 'run' is", cpd_tagwords["VB"].prob("run"))
#提取出所有标签，两两一组联系在一起
brown_tags = [tag for (tag, word) in brown_tags_words ]
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
#观察当看到出现一个词性之后看到另一个词性的概率
print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
#可以匹配一下一句话的词性连接的概率，比如I WANT TO RACE (PP VB TO PP)
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") * \
    cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") * \
    cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") * \
    cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is:", prob_tagsequence)

#实现Viterbi算法，在read me中有对该算法的具体说明
#也就是说我们现在手中有一句话，该如何知道最符合的tag是哪一组
#首先取出所有tag
distinct_tags = set(brown_tags)
sentence = ["can", "you", "hear", "me" ]
sentlen = len(sentence)
viterbi = [ ]
backpointer = [ ]
first_viterbi = { }
first_backpointer = { }
for tag in distinct_tags:
    # don't record anything for the START tag
    if tag == "START": continue
    first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
    first_backpointer[ tag ] = "START"

print(first_viterbi)
print(first_backpointer)

viterbi.append(first_viterbi)
backpointer.append(first_backpointer)
#打印出目前最好的tag
currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
print( "Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_backpointer[ currbest], currbest)

#开始做循环
for wordindex in range(1, len(sentence)):
    this_viterbi = { }
    this_backpointer = { }
    prev_viterbi = viterbi[-1]
    
    for tag in distinct_tags:
        # START没有卵用的，我们要忽略
        if tag == "START": continue
        
        # 如果现在这个tag是X，现在的单词是w，
        # 我们想找前一个tag Y，并且让最好的tag sequence以Y X结尾。
        # 也就是说
        # Y要能最大化：
        # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: \
            prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

        this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
            cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
        this_backpointer[ tag ] = best_previous
    
    # 每次找完Y 我们把目前最好的 存一下
    currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
    print( "Word", "'" + sentence[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)


    # 完结
    # 全部存下来
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)

# 找所有以END结尾的tag sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# 我们这会儿是倒着存的。。。。因为。。好的在后面
best_tagsequence = [ "END", best_previous ]
# 同理 这里也有倒过来
backpointer.reverse()

current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]
    
best_tagsequence.reverse()
print( "The sentence was:", end = " ")
for w in sentence: print( w, end = " ")
print("\n")
print( "The best tag sequence is:", end = " ")
for t in best_tagsequence: print (t, end = " ")
print("\n")
print( "The probability of the best tag sequence is:", prob_tagsequence)

#最后的结果不理想，因为语料太少，需要添加更多的语料。





