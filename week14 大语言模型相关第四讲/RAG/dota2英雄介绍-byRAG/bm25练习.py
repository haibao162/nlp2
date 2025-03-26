import json
import math
import os
import pickle
import sys
from typing import List, Dict

class BM25:
    EPSILON = 0.25
    PARAM_K1 = 1.5
    PARAM_B = 0.6

    def __init__(self, corpus: Dict):
        self.corpus_size = 0 # 文档总数
        self.wordNumsOfAllDocs = 0 # 用于计算文档集合中平均每篇文档的词数 -> wordNumsOfAllDoc / corpus_size
        self.doc_freqs = {} # 每篇文档中查询词的词频
        self.idf = {} # 记录查询词的IDF
        self.doc_len = {} # 记录每篇文档的单词数
        self.docContainedWord = {} # 包含单词word的文档集合
        self._initialize(corpus)

    def _initialize(self, corpus: Dict):
        """
            根据语料库构建倒排索引
        """
        for index, document in corpus.items():
            self.corpus_size += 1
            self.doc_len[index] = len(document) # 文档的单词数
            self.wordNumsOfAllDocs += len(document)
            
            # 一篇文档中单词出现的概率
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies
            
            # 构建词到文档的倒排索引，将包含单词的和文档和包含关系进行反向映射
            for word in frequencies.keys(): # 遍历文档中的单词
                if word not in self.docContainedWord:
                    self.docContainedWord[word] = set()
                self.docContainedWord[word].add(index)

        # 计算IDF
        idf_sum = 0
        negative_idfs = []
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word]) # 包含这个词的文档数
            idf = math.log(self.corpus_size - doc_nums_contained_word + 0.5) / (doc_nums_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        average_idf = float(idf_sum) / len(self.idf)
        eps = BM25.EPSILON * average_idf
        for word in negative_idfs:
            self.idf[word] += eps
    
    # len(D) / avg_len
    @property
    def avgdl(self):
        return float(self.wordNumsOfAllDocs) / self.corpus_size
    
    def get_score(self, query: List, doc_index):
        """
        计算查询 q 和文档 d 的相关性分数
        :param query: 查询词列表
        :param doc_index: 为语料库中某篇文档对应的索引
        """
        k1 = BM25.PARAM_K1
        b = BM25.PARAM_B
        score = 0
        doc_freqs = self.doc_freqs[doc_index]
        for word in query:
            if word not in doc_freqs:
                continue
            score += self.idf[word] * doc_freqs[word] * (k1 + 1) / (
                doc_freqs[word] + k1 * (1 - b + b * self.doc_len[doc_index] / self.avgdl)
            )
        return [doc_index, score]

    def get_scores(self, query):
        scores = [self.get_score(query, doc_index) for doc_index in range(self.doc_len.keys())]
        # scores.sort(key=lambda x: x[1], reverse=True)
        return scores

        
