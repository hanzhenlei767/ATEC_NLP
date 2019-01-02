#coding=utf-8
# -*- coding:utf-8 -*-
#/usr/bin/env python
#coding=utf-8
import jieba
import sys
#import thulac

import pickle
import json

import re

import numpy as np
import pandas as pd

from collections import defaultdict

from gensim.models import word2vec
from gensim.models import KeyedVectors

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K#返回当前后端
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding,LSTM,Layer,initializers,regularizers,constraints,Input,Dropout,concatenate,BatchNormalization
from keras.layers import Dense,Bidirectional,Concatenate,Multiply,Maximum,Subtract,Lambda,dot,Flatten,Reshape
import gc


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def load_stopwordslist(filepath):
    """
    加载停用词
    :param filepath:停用词文件路径
    :return:
    """
    with open(filepath,"r") as file:
        stop_words = [line.decode('utf-8').strip() for line in file]
        return stop_words
    
def load_spelling_corrections(filepath):
    """
    加载拼写修改词
    :param filepath:替换词文件路径
    :return:
    """
    with open(filepath,"r") as file:
        spelling_corrections = json.load(file)
        return spelling_corrections

def transform_other_word(str_text,reg_dict):
    """
    替换词
    :param str_text:待替换的句子
    :param reg_dict:替换词字典
    :return:
    """
    for token_str,replac_str in reg_dict.items():
        str_text = str_text.replace(token_str.encode('utf-8'), replac_str.encode('utf-8'))
    return str_text

def seg_sentence(sentence,stop_words):
    """
    对句子进行分词
    :param sentence:句子，停用词
    """
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ""
    for word in sentence_seged:
        if word not in stop_words:#去除停用词
            if word != " ":
                out_str += word
                out_str += " "
    return out_str


def preprocessing_word(data_df):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:
    """
    import copy

    data_processed = copy.deepcopy(data_df)

    # 加载停用词
    stopwords = load_stopwordslist(stop_words_path)
    
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(spelling_corrections_path)

    re_object = re.compile(r'\*+') #去除句子中的脱敏数字***，替换成一
    
    for index, row in data_df.iterrows():
            
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["s1", "s2"]:
            # 替换掉脱敏的数字
            re_str = re_object.subn(u"十一".encode('utf-8'),row[col_name])
            
            # 纠正一些词
            spell_corr_str = transform_other_word(re_str[0],spelling_corrections)
            
            # 分词+去除停用词
            seg_str = seg_sentence(spell_corr_str, stopwords)
            
            #分词之后的DataFrame
            data_processed.at[index, col_name] = seg_str   
    return data_processed

def preprocessing_char(data_df):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:
    """
    import copy

    data_processed = copy.deepcopy(data_df)
    # 加载停用词
    stopwords = load_stopwordslist(stop_words_path)
    
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(spelling_corrections_path)

    re_object = re.compile(r'\*+') #去除句子中的脱敏数字***，替换成一
    
    texts = []
    
    for index, row in data_processed.iterrows(): 
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["s1", "s2"]:
        	# 替换掉脱敏的数字
            re_str = re_object.subn(u"十一".encode('utf-8'),row[col_name])
            
            # 纠正一些词
            spell_corr_str = transform_other_word(re_str[0],spelling_corrections)
            spell_corr_str = spell_corr_str.decode('utf-8')
            spell_corr_str = list(spell_corr_str)

            texts.extend(spell_corr_str)        
    return texts

# train data file paths
train_data_path = 'data/atec_nlp_sim_train.csv'                 #训练数据
train_add_data_path = 'data/atec_nlp_sim_train_add.csv'         #添加训练数据

#训练数据的分析的外部数据
stop_words_path = 'data/stop_words.txt'                      	#停用词路径
tokenize_dict_path = 'data/dict_all.txt'                     	#jieba分词新自定义字典
spelling_corrections_path = 'data/spelling_corrections.json' 	#纠错及部分同义词替换规则的文件
doubt_words_path = 'data/doubt_words.txt'                    	#计算两个语句中的疑问词的相似度的疑问词相似的规则文件

# 词向量路径
train_all_wordvec_path = "data/train_all_data.bigram" 			#全部数据训练的词向量
train_all_char_wordvec_path = "data/train_all_char_data.bigram" #全部数据训练的字向量

def read_all_train_data(train_data_path,train_add_data_path,test_data_path):
	#读取训练数据
	train_data_df = pd.read_csv(train_data_path, sep='\t', header=None,names=["index", "s1", "s2", "label"])
	train_add_data_df = pd.read_csv(train_add_data_path, sep='\t', header=None, names=["index", "s1", "s2", "label"])
	test_data_df = pd.read_csv(test_data_path, sep='\t', header=None, names=["index", "s1", "s2"])

	#合并训练数据
	frames = [train_data_df, train_add_data_df]
	train_all = pd.concat(frames)
	train_all.reset_index(drop=True, inplace=True)

	#合并Word2Vec训练数据
	frames = [train_data_df, train_add_data_df,test_data_df]
	w2c_train_all = pd.concat(frames)
	w2c_train_all.reset_index(drop=True, inplace=True)

	return w2c_train_all,train_all,test_data_df


#处理数据
def preprocess_data(w2c_train_all,train_all,test_data_df):
	#加载自定义新词
	jieba.load_userdict(tokenize_dict_path)

	#预处理后的训练数据
	w2c_train_all_processed = preprocessing_word(w2c_train_all)
	train_all_processed = preprocessing_word(train_all)
	test_data_processed = preprocessing_word(test_data_df)

	return w2c_train_all_processed,train_all_processed,test_data_processed

#处理后的语料训练词向量
def pre_train_w2v(w2c_train_processed,train_all_wordvec_path,binary = False):

    texts = []
    texts_s1_train = [line.strip().split(" ") for line in w2c_train_processed['s1'].tolist()]
    texts_s2_train = [line.strip().split(" ") for line in w2c_train_processed['s2'].tolist()]
    
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)

    model = word2vec.Word2Vec(sentences=texts,size=300,window=2,min_count=3,workers=-1)
    #保存词向量
    model.wv.save_word2vec_format(train_all_wordvec_path,binary=binary,fvocab=None)

#处理后的语料训练词向量
def pre_train_char_w2v(texts,train_all_char_wordvec_path,binary = False):

    model = word2vec.Word2Vec(sentences=texts,size=300,window=3,min_count=3,workers=-1)
    #保存字向量
    model.wv.save_word2vec_format(fname=train_all_char_wordvec_path,binary=binary,fvocab=None)




# NLP特征提取


def extract_sentece_length_diff(train_all):
    """
    长度差特征
    """ 
    feature_train = np.zeros((train_all.shape[0],1),dtype='float32')

    # 计算两个句子的长度差
    def get_length_diff(s1, s2):
        return 1 - (abs(len(s1) - len(s2)) / float(max(len(s1), len(s2))))

    for index,row in train_all.iterrows():
        s1 = row['s1'].strip().split(' ')
        s2 = row['s2'].strip().split(' ')
        diff = get_length_diff(s1,s2)
        feature_train[index] = round(diff,5)

    return feature_train

def extract_edit_distance(train_all):
    """
    编辑距离特征
    """ 
    feature_train = np.zeros((train_all.shape[0], 1), dtype='float32')

    # 计算编辑距离
    def get_edit_distance(rawq1, rawq2):
        #构建DP矩阵
        m, n = len(rawq1) + 1, len(rawq2) + 1
        matrix = [[0] * n for i in range(m)]
        matrix[0][0] = 0
        for i in range(1, m):
            matrix[i][0] = matrix[i - 1][0] + 1
        for j in range(1, n):
            matrix[0][j] = matrix[0][j - 1] + 1
        cost = 0
        for i in range(1, m):
            for j in range(1, n):
                if rawq1[i - 1] == rawq2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)
        return 1 - (matrix[m - 1][n - 1] / float(max(len(rawq1), len(rawq2))))

    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        edit_distance = get_edit_distance(s1,s2)
        feature_train[index] = round(edit_distance,5)
        
    return feature_train


def extract_longest_common_substring(train_all):
    """
    公共子串特征
    """ 
    feature_train = np.zeros((train_all.shape[0], 1), dtype='float32')
    
    # 计算最长公共字符串
    def get_common_substring_len(rawq1, rawq2):
        #构建DP矩阵
        m, n = len(rawq1) + 1, len(rawq2) + 1
        matrix = [[0] * n for i in range(m)]
        longest_num = 0
        for i in range(1, m):
            for j in range(1, n):
                if rawq1[i - 1] == rawq2[j - 1]:
                    matrix[i][j] = matrix[i-1][j-1] + 1
                    if matrix[i][j] > longest_num:
                        longest_num = matrix[i][j]
                    else:
                        matrix[i][j] = 0
        return longest_num / float(min(len(rawq1), len(rawq2)))
    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        common_substring_len = get_common_substring_len(s1,s2)
        feature_train[index] = round(common_substring_len,5)
        
    return feature_train

def extract_longest_common_subsequence(train_all):
    """
    公共子序列特征
    """ 
    feature_train = np.zeros((train_all.shape[0], 1), dtype='float32')
    
    # 计算最长公共子序列
    def get_common_subsequence_len(rawq1, rawq2):
        #构建DP矩阵
        m, n = len(rawq1) + 1, len(rawq2) + 1
        matrix = [[0] * n for i in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                if rawq1[i - 1] == rawq2[j - 1]:
                    matrix[i][j] = matrix[i-1][j-1] + 1
                else:
                    matrix[i][j] = max(matrix[i-1][j],matrix[i][j-1])
        return matrix[m-1][n-1] / float(min(len(rawq1), len(rawq2)))
    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        common_subsequence_len = get_common_subsequence_len(s1,s2)
        feature_train[index] = round(common_subsequence_len,5)
        
    return feature_train


def extract_ngram(train_all,max_ngram = 3):
    '''
    提取ngram特征
    '''
    feature_train = np.zeros((train_all.shape[0], max_ngram), dtype='float32')

    # 提取n-gram词汇
    def get_ngram(rawq, ngram_value):
        result = []
        for i in range(len(rawq)):
            if i + ngram_value < len(rawq) + 1:
                result.append(rawq[i:i + ngram_value])
        return result

    #提取两个句子词的差异（归一化）
    def get_ngram_sim(q1_ngram, q2_ngram):
        q1_dict = {}
        q2_dict = {}
        
        #统计q1_ngram中个词汇的个数
        for token in q1_ngram:
            if token not in q1_dict:
                q1_dict[token] = 1
            else:
                q1_dict[token] = q1_dict[token] + 1
        #q1_ngram总词汇数
        q1_count = np.sum([value for key, value in q1_dict.items()])

        #统计q2_ngram中个词汇的个数
        for token in q2_ngram:
            if token not in q2_dict:
                q2_dict[token] = 1
            else:
                q2_dict[token] = q2_dict[token] + 1
        #q2_ngram总词汇数
        q2_count = np.sum([value for key, value in q2_dict.items()])

        # ngram1有但是ngram2没有
        q1_count_only = np.sum([value for key, value in q1_dict.items() if key not in q2_dict])
        # ngram2有但是ngram1没有
        q2_count_only = np.sum([value for key, value in q2_dict.items() if key not in q1_dict])
        # ngram1和ngram2都有的话，计算value的差值
        q1_q2_count = np.sum([abs(value - q2_dict[key]) for key, value in q1_dict.items() if key in q2_dict])
        # ngram1和ngram2的总值
        all_count = q1_count + q2_count

        return (1 - float(q1_count_only + q2_count_only + q1_q2_count) / (float(all_count) + 0.00000001))

    for ngram_value in range(max_ngram):
        for index, row in train_all.iterrows():
            s1 = row['s1'].strip()
            s2 = row['s2'].strip()
            ngram1 = get_ngram(s1, ngram_value + 1)
            ngram2 = get_ngram(s2, ngram_value + 1)
            ngram_sim = get_ngram_sim(ngram1, ngram2)
            feature_train[index,ngram_value] = round(ngram_sim,5)

    return feature_train


def extract_sentence_diff_same(train_all):
    '''
    两个句子的相同和不同的词特征
    '''
    col_num = 6
    feature_train = np.zeros((train_all.shape[0],col_num),dtype='float64')

    #统计两个句子的相同和不同
    def get_word_diff(q1, q2):
        set1 = set(q1.split(" "))
        set2 = set(q2.split(" "))
        
        #两个句子相同词的长度
        same_word_len = len(set1 & set2)
        
        #仅句子1中有的词汇个数
        unique_word1_len = len(set1 - set2)
        
        #仅句子2中有的词汇个数
        unique_word2_len = len(set2 - set1)
        
        #句子1中词汇个数
        word1_len = len(set1)
        
        #句子2中词汇个数
        word2_len = len(set2)
        
        #两句子的平均长度
        avg_len = (word1_len + word2_len) / 2.0
        
        #两个句子中较长的长度
        max_len = max(word1_len, word2_len)
        
        #两个句子中较短的长度
        min_len = min(word1_len, word2_len)
        
        #两个句子的杰卡德距离
        jaccard_sim = same_word_len / float(len(set1 | set2))

        return same_word_len / float(max_len), same_word_len / float(min_len), same_word_len / float(avg_len), \
               unique_word1_len / float(word1_len), unique_word2_len /float(word2_len), jaccard_sim

    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        features = tuple()
        features = get_word_diff(s1,s2)
        for col_index,feature in enumerate(features):
            feature_train[index,col_index] = round(feature,5)

    return feature_train


def extract_doubt_sim(train_all):
    '''
    抽取疑问词相同的比例
    '''
    feature_train = np.zeros((train_all.shape[0], 1), dtype='float32')
    
    with open(doubt_words_path,"r") as file:
        doubt_words = [line.decode('utf-8').strip() for line in file]
        
    # 获取疑问词相同的比例
    def get_doubt_sim(q1, q2, doubt_words):
        q1_doubt_words = set(q1.split(" ")) & set(doubt_words)
        q2_doubt_words = set(q2.split(" ")) & set(doubt_words)
        return len(q1_doubt_words & q2_doubt_words) / float(len(q1_doubt_words | q2_doubt_words) + 1)

    for index,row in train_all.iterrows():
        # 因为doubt_words词表加载出来的是Unicode，所以需要将s1,s2解码成Unicode
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        doubt_sim = get_doubt_sim(s1,s2,doubt_words)
        feature_train[index] = round(doubt_sim,5)

    return feature_train



def extract_sentence_exist_topic(train_all):
    """
    抽取两个句子中是否同时存在蚂蚁花呗或者蚂蚁借呗的特征,同时包含花呗为1，同时包含借呗为1，否则为0
    """
    with open(doubt_words_path,"r") as file:
        doubt_words = [line.decode('utf-8').strip() for line in file]
    feature_train = np.zeros((train_all.shape[0], 2), dtype='float32')

    def get_exist_same_topic(rawq1,rawq2):
        hua_flag = 0.
        jie_flag = 0.
        if '花呗'.decode('utf-8') in rawq1 and '花呗'.decode('utf-8') in rawq2:
            hua_flag = 1.

        if '借呗'.decode('utf-8') in rawq1 and '借呗'.decode('utf-8') in rawq2:
            jie_flag = 1.

        return hua_flag,jie_flag

    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        hua_flag, jie_flag = get_exist_same_topic(s1,s2)
        feature_train[index,0] = hua_flag
        feature_train[index,1] = jie_flag

    return feature_train


def extract_word_embedding_sim(train_all,w2v_model_path = train_all_wordvec_path):
    '''
    提取句子的词向量组合的相似度
    w2v_model_path为词向量文件
    :return:
    '''
    #定义提取特征的空间
    feature_train = np.zeros((train_all.shape[0], 2), dtype='float32')

    train_all_w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)

    # 得到句子的词向量组合（tfidf）
    def get_sen_vec(q, train_all_w2v_model, tfidf_dict, tfidf_flag=True):
        sen_vec = 0
        for word in q.split(' '):
            if word in train_all_w2v_model.vocab:
                word_vec = train_all_w2v_model.word_vec(word)
                word_tfidf = tfidf_dict.get(word, None)

                if tfidf_flag == True:
                    #tfidf有效，词向量*tfidf权重=句子向量
                    sen_vec += word_vec * word_tfidf
                else:
                    #句子向量
                    sen_vec += word_vec
        sen_vec = sen_vec / np.sqrt(np.sum(np.power(sen_vec, 2)) + 0.000001)
        return sen_vec

    def get_sentece_embedding_sim(q1, q2, train_all_w2v_model, tfidf_dict, tfidf_flag=True):
        # 得到两个问句的词向量组合
        q1_sec = get_sen_vec(q1, train_all_w2v_model, tfidf_dict, tfidf_flag)
        q2_sec = get_sen_vec(q2, train_all_w2v_model, tfidf_dict, tfidf_flag)

        # 曼哈顿距离
        # manhattan_distance = np.sum(np.abs(np.subtract(q1_sec, q2_sec)))

        # 欧式距离
        enclidean_distance = np.sqrt(np.sum(np.power((q1_sec - q2_sec),2)))

        # 余弦相似度
        molecular = np.sum(np.multiply(q1_sec, q2_sec))
        denominator = np.sqrt(np.sum(np.power(q1_sec, 2))) * np.sqrt(np.sum(np.power(q2_sec, 2)))
        cos_sim = molecular / (denominator + 0.000001)

        # 闵可夫斯基距离
        # minkowski_distance = np.power(np.sum(np.power(np.abs(np.subtract(q1_sec, q2_sec)), 3)), 0.333333)

        return enclidean_distance, cos_sim
        #return cos_sim

    for index,row in train_all.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        sen_enclidean_sim,sen_cos_sim = get_sentece_embedding_sim(s1,s2,train_all_w2v_model,{},False)
        #feature_train[index,0] = round(sen_manhattan_sim,5)
        feature_train[index,0] = round(sen_enclidean_sim,5)
        feature_train[index,1] = round(sen_cos_sim,5)
        #feature_train[index,3] = round(sen_minkowski_sim,5)
    return feature_train





# 深度学习特征提取

def precision(y_true, y_pred):
	'''
	查准率（精确率）
	'''
	y_t = y_true
	y_p = y_pred

	true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_p, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	'''
	查全率（召回率）
	'''
	y_t = y_true
	y_p = y_pred

	true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_t, 0, 1)))

	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def fbeta_score(y_t, y_p, beta=1):
	'''
	Fbeta score
	'''
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_t, 0, 1))) == 0:
		return 0
	p = precision(y_t, y_p)
	r = recall(y_t, y_p)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score




def process_embedding_wv(data_all_processed,w2v_model_path = train_all_wordvec_path):

	embedding_size = 300#词向量维度
	max_sentence_length = 20 #最大句子长度

	#数据准备
	texts_s1_train = data_all_processed['s1'].tolist()
	texts_s2_train = data_all_processed['s2'].tolist()

	#合并文本数据
	texts = []
	texts.extend(texts_s1_train)
	texts.extend(texts_s2_train)

	#文本处理    
	tokenizer = Tokenizer(
		num_words=100000,
		split=' ',
		lower=False,
		char_level=False,
		filters=''
	)

	# 生成各个词对应的index列表
	tokenizer.fit_on_texts(texts)

	# 将文章以index表示
	s1_train_ids = tokenizer.texts_to_sequences(texts_s1_train)
	s2_train_ids = tokenizer.texts_to_sequences(texts_s2_train)

	#将文章以矩阵的形式（长度多退少补）保存
	s1_train_ids_pad = sequence.pad_sequences(s1_train_ids,maxlen=max_sentence_length)
	s2_train_ids_pad = sequence.pad_sequences(s2_train_ids,maxlen=max_sentence_length)

	#词序列(word_index：key:词，value:索引（编号）)
	word_index_dict = tokenizer.word_index

	# 训练集的词汇表的词向量矩阵,行数为最大值+1,形式为：index->vec
	embedding_matrix = 1 * np.random.randn(len(word_index_dict) + 1, embedding_size)
	embedding_matrix[0] = np.random.randn(embedding_size)

	# 加载预训练的词向量w2v
	w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)

	for word,index in word_index_dict.items():
		if word in w2v_model.vocab:
			embedding_matrix[index] = w2v_model.word_vec(word)

	return embedding_matrix,s1_train_ids_pad,s2_train_ids_pad

def process_char_embedding_wv(w2c_train_all,w2v_model_path = train_all_char_wordvec_path):
	embedding_size = 300#词向量维度
	max_sentence_length = 20 #最大句子长度

	#数据准备
	texts_s1_train = w2c_train_all['s1'].tolist()
	texts_s2_train = w2c_train_all['s2'].tolist()

	#合并文本数据
	texts = []
	texts.extend(texts_s1_train)
	texts.extend(texts_s2_train)

	#文本处理    
	tokenizer = Tokenizer(
		num_words=100000,
		split=' ',
		lower=True,
		char_level=True,
		filters=''
	)

	# 生成各个词对应的index列表
	tokenizer.fit_on_texts(texts)

	# 将文章以index表示
	s1_train_ids = tokenizer.texts_to_sequences(texts_s1_train)
	s2_train_ids = tokenizer.texts_to_sequences(texts_s2_train)

	#将文章以矩阵的形式（长度多退少补）保存
	s1_train_ids_pad = sequence.pad_sequences(s1_train_ids,maxlen=max_sentence_length)
	s2_train_ids_pad = sequence.pad_sequences(s2_train_ids,maxlen=max_sentence_length)

	#词序列(word_index：key:词，value:索引（编号）)
	word_index_dict = tokenizer.word_index

	# 训练集的词汇表的词向量矩阵,行数为最大值+1,形式为：index->vec
	embedding_char_matrix = 1 * np.random.randn((len(word_index_dict) + 1), embedding_size)
	embedding_char_matrix[0] = np.random.randn(embedding_size)

	w2v_char_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)

	for char,index in word_index_dict.items():
		if char in w2v_char_model.vocab:
			embedding_char_matrix[index] = w2v_char_model.word_vec(char)

	#总共有n个字，在模型里有m个词(Word2Vec会把低频词过滤掉，默认是5个，可通过min_count设置)
	return embedding_char_matrix,s1_train_ids_pad,s2_train_ids_pad


class ConsDist(Layer):
    """
    自定义定义曼哈顿距离计算层，继承Layer层，必须实现三个父类方法
    build,call,comput_output_shape
    """
    def __init__(self, **kwargs):
        self.res = None  # 表示相似度
        # self.match_vector = None
        super(ConsDist, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
              """
        super(ConsDist, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        # 计算曼哈顿距离,因为输入计算曼哈顿距离的有两个Input层分别为inputs[0]和inputs[1]
        # lstm model
        self.res = K.sum(inputs[0] * inputs[1],axis=1,keepdims=True)/(K.sum(inputs[0]**2,axis=1,keepdims=True) * K.sum(inputs[1]**2,axis=1,keepdims=True))
        return self.res
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.res)



class ManDist(Layer):
    """
    自定义定义曼哈顿距离计算层，继承Layer层，必须实现三个父类方法
    build,call,comput_output_shape
    """
    def __init__(self, **kwargs):
        self.res = None  # 表示相似度
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
        """
        super(ManDist, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        # 计算曼哈顿距离,因为输入计算曼哈顿距离的有两个Input层分别为inputs[0]和inputs[1]
        self.res  = K.exp(- K.sum(K.abs(inputs[0]-inputs[1]),axis = 1,keepdims = True))
        return self.res

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.res)


class AttentionLayer1(Layer):
    def __init__(self, **kwargs):
        # self.res = None  # 表示相似度
        self.match_vector = None
        super(AttentionLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.
              # Arguments
                  input_shape: Keras tensor (future input to layer)
                      or list/tuple of Keras tensors to reference
                      for weight shape computations.
              """
        super(AttentionLayer1, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
         # Arguments
             inputs: Input tensor, or list/tuple of input tensors.
             **kwargs: Additional keyword arguments.
         # Returns
             A tensor or list/tuple of tensors.
         """
        encode_s1 = inputs[0]
        encode_s2 = inputs[1]
        sentence_differerce = encode_s1 - encode_s2
        sentece_product = encode_s1 * encode_s2
        self.match_vector = K.concatenate([encode_s1,sentence_differerce,sentece_product,encode_s2],1)
        #
        return self.match_vector

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
               Assumes that the layer will be built
               to match that input shape provided.
               # Arguments
                   input_shape: Shape tuple (tuple of integers)
                       or list of shape tuples (one per output tensor of the layer).
                       Shape tuples can include None for free dimensions,
                       instead of an integer.

               # Returns
                   An input shape tuple.
               """
        return K.int_shape(self.match_vector)

class AttentionLayer(Layer):
    def __init__(self,step_dim,W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(AttentionLayer,self).__init__(**kwargs)#用于调用父类(超类)的一个方法。

    #设置self.supports_masking = True后需要复写该方法
    def compute_mask(self, inputs, mask=None):
        return None

    #参数设置，必须实现
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    # input (None,sentence_length,embedding_size)
    def call(self, x, mask = None):
        # 计算输出
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



#模型一：基于LSTM进行语义编码的曼哈顿距离相似度和余弦相似度的模型


def create_siamese_lstm_ManDistance_model(embedding_matrix,embedding_size = 300,max_sentence_length = 20):
	# 定义孪生网络的公共层
	X = Sequential()
	embedding_layer = Embedding(
		input_dim=len(embedding_matrix,),
		output_dim=embedding_size,
		weights=[embedding_matrix],
		trainable=True,
		input_length=max_sentence_length
	)

	lstm_layer = LSTM(
		units=50,
		dropout=0.,
		recurrent_dropout=0.,
		return_sequences=False
	)

	X.add(embedding_layer)
	X.add(lstm_layer)

	#share_model为孪生网络的共同拥有的层
	share_model = X

	# 模型是多输入的结构，定义两个句子的输入
	left_input = Input(shape=(max_sentence_length,), dtype='int32')
	right_input = Input(shape=(max_sentence_length,), dtype='int32')

	# 定义两个输入合并后的模型层
	s1_net = share_model(left_input)
	s2_net = share_model(right_input)

	# 定义输出层
	man_layer = ManDist()([s1_net,s2_net])

	out_put_layer = Dense(2, activation='softmax')(man_layer)
	# out_put_layer = Dense(1,activation='sigmoid')(man_layer)

	model = Model(inputs=[left_input, right_input],outputs=[out_put_layer], name="simaese_lstm_manDist")
	model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=["accuracy",fbeta_score,precision,recall])
	model.summary()
	return model





def extract_feature_siamese_lstm_manDist(train_all,w2v_model_path = train_all_wordvec_path):
	'''
	深度学习：提取句子的曼哈顿距离相似度
	w2v_model_path为词向量文件
	:return:
	'''
	#配置参数
	RANOD_SEED = 42
	np.random.seed(RANOD_SEED)
	nepoch = 15
	num_folds = 3
	batch_size = 1024
	embedding_size = 300#词向量维度
	max_sentence_length = 20

	#保存标签
	y_train = train_all["label"].tolist()
	y_train = np.array(y_train)
	
	embedding_matrix,X_train_s1,X_train_s2 = process_embedding_wv(train_all,w2v_model_path)

	#标签：y_train
	#输入：s1_train_ids_pad，s2_train_ids_pad
	#嵌入层矩阵：embedding_matrix

	kfold = StratifiedKFold(
		n_splits=num_folds,
		shuffle=True,
		random_state=RANOD_SEED
	)
	# 存放最后预测结果
	y_train_oofp = np.zeros((len(y_train),2),dtype='float32')

	#softmax的标签
	label = to_categorical(y_train, 2)

	for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1,y_train)):
	#对于正例的语句对样本数量少的问题，通过将正例的样本语句对进行顺序调换，形成新的正例样本对。

		# 提取训练集中的正样本和标签
		train_true_mask = y_train[ix_train] == 1                    #选出训练集的正样本标签的索引
		X_train_true_s1 = X_train_s1[ix_train][train_true_mask]     #选出s1训练样本的 正样本 
		X_train_true_s2 = X_train_s2[ix_train][train_true_mask]     #选出s2训练样本的 正样本
		y_train_true = label[ix_train][train_true_mask]             #选出训练集的正样本标签

		# 将训练集 和 训练集的正样本（s1和s2调换位置）
		X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train],X_train_true_s2])#合并训练集s1 和 训练集s2的正样本
		X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train],X_train_true_s1])#合并训练集s2 和 训练集s1的正样本
		y_add_train_fold = np.concatenate([label[ix_train],y_train_true])      #合并训练集标签标签 和 训练集的正样本标签

		# 选出验证集中的正样本和标签
		val_true_mask = y_train[ix_val]==1                          #选出验证集的正样本标签的索引
		X_val_true_s1 = X_train_s1[ix_val][val_true_mask]           #选出s1验证样本的 正样本
		X_val_true_s2 = X_train_s2[ix_val][val_true_mask]           #选出s2验证样本的 正样本
		y_val_true = label[ix_val][val_true_mask]                   #选出训验证的正样本标签

		# 将验证集 和 验证集的正样本（s1和s2调换位置）
		X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])#合并验证集s1 和 验证集s2的正样本
		X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])#合并验证集s2 和 验证集s1的正样本
		y_add_val_fold = np.concatenate([label[ix_val], y_val_true])      #合并验证集标签标签 和 验证集的正样本标签

		#打印训练的是5折的第几折
		print ('start train fold {} of {} ......'.format((fold_num + 1), 5))

		# 创建模型
		model = create_siamese_lstm_ManDistance_model(embedding_matrix)

		# 训练模型
		model_checkpoint_path = 'dl_siamese_lstm_manDist_model{}.h5'.format(fold_num)


		model.fit(x=[X_add_train_fold_s1,X_add_train_fold_s2],y=y_add_train_fold,
						validation_data=([X_add_val_fold_s1,X_add_val_fold_s2],y_add_val_fold),
						batch_size=batch_size,
						epochs=nepoch,
						verbose=1,
						class_weight={0: 1, 1: 2},
						callbacks=[
						EarlyStopping(
							monitor='val_loss',  #监控的方式：’acc’,’val_acc’,’loss’,’val_loss’
							min_delta=0.005,     #增大或者减小的阈值，只有只有大于这个部分才算作improvement
							patience=2,          #连续n次没有提升
							verbose=1,           #信息展示模式
							mode='auto'          #‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
						),
						ModelCheckpoint(
							model_checkpoint_path,
							monitor='val_loss',
							save_best_only=True,
							save_weights_only=False,
							verbose=1
						)]
					)

		model.load_weights(model_checkpoint_path)

		y_train_oofp[ix_val] = model.predict([X_train_s1[ix_val],X_train_s2[ix_val]]) #两列的概率
		
		K.clear_session()

		del X_add_train_fold_s1
		del X_add_train_fold_s2
		del X_add_val_fold_s1
		del X_add_val_fold_s2
		del y_add_train_fold
		del y_add_val_fold

		gc.collect()

	return y_train_oofp


def extract_feature_siamese_lstm_manDist_test(test_all,w2v_model_path = train_all_wordvec_path):

	#保存标签
	embedding_matrix,X_test_s1,X_test_s2 = process_embedding_wv(test_all,w2v_model_path)

	# save feature

	model_path = 'dl_siamese_lstm_manDist_model0.h5'

	model0 = load_model(model_path,custom_objects={'ManDist': ManDist, 'fbeta_score': fbeta_score, 'precision': precision, 'recall': recall})
	
	y_test_oofp = model0.predict([X_test_s1,X_test_s2])

	return y_test_oofp




#模型二：基于LSTM进行语义编码的match vector形式计算的相似度的模型


def create_siamese_lstm_attention_model(embedding_matrix,embedding_size = 300,max_sentence_length = 20):
    # 定义孪生网络的公共层
    X = Sequential()
    embedding_layer = Embedding(
        input_dim=len(embedding_matrix,),
        output_dim=embedding_size,
        weights=[embedding_matrix],
        trainable=True,
        input_length=max_sentence_length
    )

    lstm_layer = LSTM(
        units=50,
        return_sequences=False
    )

    X.add(embedding_layer)
    X.add(lstm_layer)

    #share_model为孪生网络的共同拥有的层
    share_model = X

    # 模型是多输入的结构，定义两个句子的输入
    left_input = Input(shape=(max_sentence_length,), dtype='int32')
    right_input = Input(shape=(max_sentence_length,), dtype='int32')

    # 定义两个输入合并后的模型层
    s1_net = share_model(left_input)
    s2_net = share_model(right_input)


    matching_layer = AttentionLayer1()([s1_net,s2_net])

    merge_model = Dense(128)(matching_layer)#num_dense：128
    merge_model = Dropout(0.75)(merge_model)#desen_dropout_rate：0.75
    merge_model = BatchNormalization()(merge_model)

    # 定义输出层
    output_layer = Dense(1,activation='sigmoid')(merge_model)

    model = Model(inputs=[left_input, right_input],outputs=[output_layer], name="simaese_lstm_attention")
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy",fbeta_score,precision,recall])
    model.summary()
    return model


def extract_feature_siamese_lstm_attention(train_all,w2v_model_path = train_all_wordvec_path):
    '''
    深度学习：提取句子的曼哈顿距离相似度
    w2v_model_path为词向量文件
    :return:
    '''
    #配置参数
    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 15
    num_folds = 3
    batch_size = 1024
    embedding_size = 300#词向量维度
    max_sentence_length = 20

    #保存标签
    y_train = train_all["label"].tolist()
    y_train = np.array(y_train)
    
    embedding_matrix,X_train_s1,X_train_s2 = process_embedding_wv(train_all,w2v_model_path)

    #标签：y_train
    #输入：s1_train_ids_pad，s2_train_ids_pad
    #嵌入层矩阵：embedding_matrix

    kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANOD_SEED
    )
    # 存放最后预测结果
    y_train_oofp = np.zeros((len(y_train),1),dtype='float32')

    #softmax的标签
    #label = to_categorical(y_train, 2)

    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1,y_train)):
    #对于正例的语句对样本数量少的问题，通过将正例的样本语句对进行顺序调换，形成新的正例样本对。

        # 提取训练集中的正样本和标签
        train_true_mask = y_train[ix_train] == 1                    #选出训练集的正样本标签的索引
        X_train_true_s1 = X_train_s1[ix_train][train_true_mask]     #选出s1训练样本的 正样本 
        X_train_true_s2 = X_train_s2[ix_train][train_true_mask]     #选出s2训练样本的 正样本
        y_train_true = y_train[ix_train][train_true_mask]             #选出训练集的正样本标签

        # 将训练集 和 训练集的正样本（s1和s2调换位置）
        X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train],X_train_true_s2])#合并训练集s1 和 训练集s2的正样本
        X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train],X_train_true_s1])#合并训练集s2 和 训练集s1的正样本
        y_add_train_fold = np.concatenate([y_train[ix_train],y_train_true])      #合并训练集标签标签 和 训练集的正样本标签

        # 选出验证集中的正样本和标签
        val_true_mask = y_train[ix_val]==1                          #选出验证集的正样本标签的索引
        X_val_true_s1 = X_train_s1[ix_val][val_true_mask]           #选出s1验证样本的 正样本
        X_val_true_s2 = X_train_s2[ix_val][val_true_mask]           #选出s2验证样本的 正样本
        y_val_true = y_train[ix_val][val_true_mask]                   #选出训验证的正样本标签

        # 将验证集 和 验证集的正样本（s1和s2调换位置）
        X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])#合并验证集s1 和 验证集s2的正样本
        X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])#合并验证集s2 和 验证集s1的正样本
        y_add_val_fold = np.concatenate([y_train[ix_val], y_val_true])      #合并验证集标签标签 和 验证集的正样本标签

        #打印训练的是5折的第几折
        print ('start train fold {} of {} ......'.format((fold_num + 1), 5))

        # 创建模型
        model = create_siamese_lstm_attention_model(embedding_matrix)

        # 训练模型
        model_checkpoint_path = 'dl_siamese_lstm_attention_model{}.h5'.format(fold_num)


        model.fit(x=[X_add_train_fold_s1,X_add_train_fold_s2],y=y_add_train_fold,
                        validation_data=([X_add_val_fold_s1,X_add_val_fold_s2],y_add_val_fold),
                        batch_size=batch_size,
                        epochs=nepoch,
                        verbose=1,
                        class_weight={0: 1, 1: 2},
                        callbacks=[
                        EarlyStopping(
                            monitor='val_loss',  #监控的方式：’acc’,’val_acc’,’loss’,’val_loss’
                            min_delta=0.005,     #增大或者减小的阈值，只有只有大于这个部分才算作improvement
                            patience=2,          #连续n次没有提升
                            verbose=1,           #信息展示模式
                            mode='auto'          #‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
                        ),
                        ModelCheckpoint(
                            model_checkpoint_path,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=False,
                            verbose=1
                        )]
                    )

        model.load_weights(model_checkpoint_path)

        y_train_oofp[ix_val] = model.predict([X_train_s1[ix_val],X_train_s2[ix_val]]) #两列的概率
        
        K.clear_session()

        del X_add_train_fold_s1
        del X_add_train_fold_s2
        del X_add_val_fold_s1
        del X_add_val_fold_s2
        del y_add_train_fold
        del y_add_val_fold

        gc.collect()

    return y_train_oofp


def extract_feature_siamese_lstm_attention_test(test_all,w2v_model_path = train_all_wordvec_path):

    #保存标签
    embedding_matrix,X_test_s1,X_test_s2 = process_embedding_wv(test_all,w2v_model_path)

    # save feature

    model_path = 'dl_siamese_lstm_attention_model0.h5'

    model0 = load_model(model_path,custom_objects={'AttentionLayer1': AttentionLayer1, 'fbeta_score': fbeta_score, 'precision': precision, 'recall': recall})
    
    y_test_oofp = model0.predict([X_test_s1,X_test_s2])

    return y_test_oofp





#模型三：改进的Compare-Aggregate的模型


def create_siamese_lstm_dssm_mdoel(embedding_matrix,embedding_word_matrix,embedding_size = 300,max_sentence_length = 20,max_word_length=20):
    # 第一部分
    # step 1 定义复杂模型的输入
    num_conv2d_layers = 1
    filters_2d = [6, 12]
    kernel_size_2d = [[3, 3], [3, 3]]
    mpool_size_2d = [[2, 2], [2, 2]]

    left_input = Input(shape=(max_sentence_length,), dtype='int32')
    right_input = Input(shape=(max_sentence_length,), dtype='int32')

    # 定义需要使用的网络层
    embedding_layer1 = Embedding(
        input_dim=len(embedding_matrix, ),
        output_dim=embedding_size,
        weights=[embedding_matrix],
        trainable=True,
        input_length=max_sentence_length
    )
    att_layer1 = AttentionLayer(20)
    bi_lstm_layer =Bidirectional(LSTM(50))
    lstm_layer1 = LSTM(50,return_sequences=True)#return_sequences：返回全部time step 的 hidden state值
    lstm_layer2 = LSTM(50)

    # 组合模型结构,两个输入添加Embeding层
    s1 = embedding_layer1(left_input)
    s2 = embedding_layer1(right_input)

    # 在Embeding层上添加双向LSTM层
    s1_bi = bi_lstm_layer(s1)
    s2_bi = bi_lstm_layer(s2)

    # 另在Embeding层上添加双层LSTM层
    s1_lstm_lstm = lstm_layer2(lstm_layer1(s1))
    s2_lstm_lstm = lstm_layer2(lstm_layer1(s2))

    s1_lstm = lstm_layer1(s1)
    s2_lstm = lstm_layer1(s2)

    cnn_input_layer = dot([s1_lstm,s2_lstm],axes=-1)
    cnn_input_layer_dot = Reshape((20,20,-1))(cnn_input_layer)
    layer_conv1 = Conv2D(filters=8,kernel_size=3,padding='same',activation='relu')(cnn_input_layer_dot)
    z = MaxPooling2D(pool_size=(2,2))(layer_conv1)

    for i in range(num_conv2d_layers):
        z = Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same', activation='relu')(z)
        z = MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)

    pool1_flat = Flatten()(z)
    # # print pool1_flat
    pool1_flat_drop = Dropout(rate=0.1)(pool1_flat)
    ccn1 = Dense(32, activation='relu')(pool1_flat_drop)
    ccn2 = Dense(16, activation='relu')(ccn1)

    # 另在Embeding层上添加attention层
    s1_att = att_layer1(s1)
    s2_att = att_layer1(s2)

    # 组合在Embeding层上添加attention层和在Embeding层上添加双向LSTM层
    s1_last = Concatenate(axis=1)([s1_att,s1_bi])
    s2_last = Concatenate(axis=1)([s2_att,s2_bi])

    cos_layer = ConsDist()([s1_last,s2_last])
    man_layer = ManDist()([s1_last,s2_last])
    # 第二部分
    left_w_input = Input(shape=(max_word_length,), dtype='int32')
    right_w_input = Input(shape=(max_word_length,), dtype='int32')

    # 定义需要使用的网络层
    embedding_layer2 = Embedding(
        input_dim=len(embedding_word_matrix, ),
        output_dim=embedding_size,
        weights=[embedding_word_matrix],
        trainable=True,
        input_length=max_word_length
    )
    lstm_word_bi_layer = Bidirectional(LSTM(6))
    att_layer2 = AttentionLayer(20)

    s1_words = embedding_layer2(left_w_input)
    s2_words = embedding_layer2(right_w_input)

    s1_words_bi = lstm_word_bi_layer(s1_words)
    s2_words_bi = lstm_word_bi_layer(s2_words)

    s1_words_att = att_layer2(s1_words)
    s2_words_att = att_layer2(s2_words)

    s1_words_last = Concatenate(axis=1)([s1_words_att,s1_words_bi])
    s2_words_last = Concatenate(axis=1)([s2_words_att,s2_words_bi])
    cos_layer1 = ConsDist()([s1_words_last,s2_words_last])
    man_layer1 = ManDist()([s1_words_last,s2_words_last])


    # 第三部分，前两部分模型组合
    s1_s2_mul = Multiply()([s1_last,s2_last])
    s1_s2_sub = Lambda(lambda x: K.abs(x))(Subtract()([s1_last,s2_last]))
    s1_s2_maxium = Maximum()([Multiply()([s1_last,s1_last]),Multiply()([s2_last,s2_last])])
    s1_s2_sub1 = Lambda(lambda x: K.abs(x))(Subtract()([s1_lstm_lstm,s2_lstm_lstm]))


    s1_words_s2_words_mul = Multiply()([s1_words_last,s2_words_last])
    s1_words_s2_words_sub = Lambda(lambda x: K.abs(x))(Subtract()([s1_words_last,s2_words_last]))
    s1_words_s2_words_maxium = Maximum()([Multiply()([s1_words_last,s1_words_last]),Multiply()([s2_words_last,s2_words_last])])

    last_list_layer = Concatenate(axis=1)([s1_s2_mul,s1_s2_sub,s1_s2_sub1,s1_s2_maxium,s1_words_s2_words_mul,s1_words_s2_words_sub,s1_words_s2_words_maxium])
    last_list_layer = Dropout(0.05)(last_list_layer)
    # Dense 层
    dense_layer1 = Dense(32,activation='relu')(last_list_layer)
    dense_layer2 = Dense(48,activation='sigmoid')(last_list_layer)

    output_layer = Concatenate(axis=1)([dense_layer1,dense_layer2,cos_layer,man_layer,cos_layer1,man_layer1,ccn2])
    # Step4 定义输出层
    output_layer = Dense(1, activation='sigmoid')(output_layer)

    model = Model(inputs=[left_input,right_input,left_w_input,right_w_input],outputs=[output_layer], name="simaese_lstm_attention")
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy", fbeta_score, precision, recall])
    return model

def extract_feature_siamese_lstm_dssm(train_all,train_all_processed,w2v_model_path = train_all_wordvec_path,w2v_char_model_path = train_all_char_wordvec_path):
    '''
    深度学习：提取
    w2v_model_path为词向量文件
    :return:
    '''
    #配置参数
    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 3
    num_folds = 2
    batch_size = 1024
    embedding_size = 300#词向量维度
    max_sentence_length = 20

    #保存标签
    y_train = train_all["label"].tolist()
    y_train = np.array(y_train)
    
    embedding_matrix,X_train_s1,X_train_s2 = process_embedding_wv(train_all_processed,w2v_model_path)
    embedding_char_matrix,X_char_train_s1,X_char_train_s2 = process_char_embedding_wv(train_all,w2v_char_model_path)
    #标签：y_train
    #输入：s1_train_ids_pad，s2_train_ids_pad
    #嵌入层矩阵：embedding_matrix

    kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANOD_SEED
    )
    # 存放最后预测结果
    y_train_oofp = np.zeros((len(y_train),1),dtype='float32')

    #softmax的标签
    #label = to_categorical(y_train, 2)

    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1,y_train)):
    #对于正例的语句对样本数量少的问题，通过将正例的样本语句对进行顺序调换，形成新的正例样本对。

        # 提取训练集中的正样本和标签
        train_true_mask = y_train[ix_train] == 1                    #选出训练集的正样本标签的索引
        X_train_true_s1 = X_train_s1[ix_train][train_true_mask]     #选出s1训练样本的 正样本 
        X_train_true_s2 = X_train_s2[ix_train][train_true_mask]     #选出s2训练样本的 正样本
        y_train_true = y_train[ix_train][train_true_mask]            #选出训练集的正样本标签

        # 将训练集 和 训练集的正样本（s1和s2调换位置）
        X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train],X_train_true_s2])#合并训练集s1 和 训练集s2的正样本
        X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train],X_train_true_s1])#合并训练集s2 和 训练集s1的正样本
        y_add_train_fold = np.concatenate([y_train[ix_train],y_train_true])      #合并训练集标签标签 和 训练集的正样本标签

        X_train_true_s1_char = X_char_train_s1[ix_train][train_true_mask]
        X_train_true_s2_char = X_char_train_s2[ix_train][train_true_mask]

        # 进行添加
        X_add_train_fold_s1_char = np.vstack([X_char_train_s1[ix_train], X_train_true_s2_char])
        X_add_train_fold_s2_char = np.vstack([X_char_train_s2[ix_train], X_train_true_s1_char])

        # 选出验证集中的正样本和标签
        val_true_mask = y_train[ix_val]==1                          #选出验证集的正样本标签的索引
        X_val_true_s1 = X_train_s1[ix_val][val_true_mask]           #选出s1验证样本的 正样本
        X_val_true_s2 = X_train_s2[ix_val][val_true_mask]           #选出s2验证样本的 正样本
        y_val_true = y_train[ix_val][val_true_mask]                   #选出训验证的正样本标签

        # 将验证集 和 验证集的正样本（s1和s2调换位置）
        X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])#合并验证集s1 和 验证集s2的正样本
        X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])#合并验证集s2 和 验证集s1的正样本
        y_add_val_fold = np.concatenate([y_train[ix_val], y_val_true])      #合并验证集标签标签 和 验证集的正样本标签

        X_val_true_s1_char = X_char_train_s1[ix_val][val_true_mask]
        X_val_true_s2_char = X_char_train_s2[ix_val][val_true_mask]

        X_add_val_fold_s1_char = np.vstack([X_char_train_s1[ix_val], X_val_true_s2_char])
        X_add_val_fold_s2_char = np.vstack([X_char_train_s2[ix_val], X_val_true_s1_char])
        
        #打印训练的是5折的第几折
        print ('start train fold {} of {} ......'.format((fold_num + 1), 5))

        # 创建模型
        model = create_siamese_lstm_dssm_mdoel(embedding_matrix,embedding_char_matrix)

        # 训练模型
        model_checkpoint_path = 'dl_siamese_lstm_dssm_model{}.h5'.format(fold_num)

        model.fit(x=[X_add_train_fold_s1, X_add_train_fold_s2,X_add_train_fold_s1_char,X_add_train_fold_s2_char], y=y_add_train_fold,
                  validation_data=([X_add_val_fold_s1, X_add_val_fold_s2,X_add_val_fold_s1_char,X_add_val_fold_s2_char], y_add_val_fold),
                  batch_size=batch_size,
                  epochs=nepoch,
                  class_weight={0:1,1:2},
                  verbose=1,
                  callbacks=[
                      EarlyStopping(
                          monitor='val_loss',
                          min_delta=0.001,
                          patience=3,
                          verbose=1,
                          mode='auto'
                      ),
                      ModelCheckpoint(
                          model_checkpoint_path,
                          monitor='val_loss',
                          save_best_only=True,
                          save_weights_only=False,
                          verbose=1
                      )]
                  )
        model.load_weights(model_checkpoint_path)

        y_train_oofp[ix_val] = model.predict([X_train_s1[ix_val], X_train_s2[ix_val],X_char_train_s1[ix_val],X_char_train_s2[ix_val]])
        
        K.clear_session()

        del X_add_train_fold_s1
        del X_add_train_fold_s2
        del X_add_val_fold_s1
        del X_add_val_fold_s2
        del y_add_train_fold
        del y_add_val_fold

        gc.collect()

    return y_train_oofp


def extract_feature_siamese_lstm_dssm_test(test_all,test_all_processed,w2v_model_path = train_all_wordvec_path,w2v_char_model_path = train_all_char_wordvec_path):

    #保存标签
    embedding_matrix,X_test_s1,X_test_s2 = process_embedding_wv(test_all_processed,w2v_model_path)

    embedding_char_matrix,X_char_test_s1,X_char_test_s2 = process_char_embedding_wv(test_all,w2v_char_model_path)
    # save feature

    model_path = 'dl_siamese_lstm_dssm_model0.h5'

    model0 = load_model(model_path,custom_objects={'AttentionLayer': AttentionLayer, 'fbeta_score': fbeta_score, 'precision': precision, 'recall': recall})
    
    y_test_oofp = model0.predict([X_test_s1, X_test_s2,X_char_test_s1,X_char_test_s2])

    return y_test_oofp

def extract_feature(train_all,test_all,train_all_processed,test_all_processed):
	#NLP特征
	sentece_length_diff_feature = extract_sentece_length_diff(train_all_processed)
	edit_distance_feature = extract_edit_distance(train_all_processed)
	longest_common_substring_feature = extract_longest_common_substring(train_all_processed)
	longest_common_subsequence_feature = extract_longest_common_subsequence(train_all_processed)
	ngram_feature = extract_ngram(train_all_processed)
	sentence_diff_same_feature = extract_sentence_diff_same(train_all_processed)
	doubt_sim_feature = extract_doubt_sim(train_all_processed)
	sentence_exist_topic_feature = extract_sentence_exist_topic(train_all_processed)
	word_embedding_sim_feature = extract_word_embedding_sim(train_all_processed)
	#DL特征
	#siamese_lstm_manDist_feature = extract_feature_siamese_lstm_manDist(train_all_processed,train_all_wordvec_path)
	#siamese_lstm_attention_feature = extract_feature_siamese_lstm_attention(train_all_processed,train_all_wordvec_path)
	siamese_lstm_dssm_feature = extract_feature_siamese_lstm_dssm(train_all,train_all_processed,train_all_wordvec_path,train_all_char_wordvec_path)
	#合并特征
	X_train = np.concatenate([sentece_length_diff_feature,
						edit_distance_feature,
						longest_common_substring_feature,
						longest_common_subsequence_feature,
						ngram_feature,
						sentence_diff_same_feature,
						doubt_sim_feature,
						sentence_exist_topic_feature,
						word_embedding_sim_feature,
						#siamese_lstm_manDist_feature,
						#siamese_lstm_attention_feature,
						siamese_lstm_dssm_feature],
						axis = 1)
	#NLP特征
	sentece_length_diff_feature = extract_sentece_length_diff(test_all_processed)
	edit_distance_feature = extract_edit_distance(test_all_processed)
	longest_common_substring_feature = extract_longest_common_substring(test_all_processed)
	longest_common_subsequence_feature = extract_longest_common_subsequence(test_all_processed)
	ngram_feature = extract_ngram(test_all_processed)
	sentence_diff_same_feature = extract_sentence_diff_same(test_all_processed)
	doubt_sim_feature = extract_doubt_sim(test_all_processed)
	sentence_exist_topic_feature = extract_sentence_exist_topic(test_all_processed)
	word_embedding_sim_feature = extract_word_embedding_sim(test_all_processed)
	#DL特征
	#siamese_lstm_manDist_feature = extract_feature_siamese_lstm_manDist_test(test_all_processed,train_all_wordvec_path)
	#siamese_lstm_attention_feature = extract_feature_siamese_lstm_attention_test(test_all_processed,train_all_wordvec_path)
	siamese_lstm_dssm_feature = extract_feature_siamese_lstm_dssm_test(test_all,test_all_processed,train_all_wordvec_path,train_all_char_wordvec_path)
	#合并特征
	X_test = np.concatenate([sentece_length_diff_feature,
						edit_distance_feature,
						longest_common_substring_feature,
						longest_common_subsequence_feature,
						ngram_feature,
						sentence_diff_same_feature,
						doubt_sim_feature,
						sentence_exist_topic_feature,
						word_embedding_sim_feature,
						#siamese_lstm_manDist_feature,
						#siamese_lstm_attention_feature,
						siamese_lstm_dssm_feature],
						axis = 1)
	return X_train,X_test


#stacking 模型训练


#################### Stacking 模型的融合 ####################
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class StackingBaseClassifier(object):

    def train(self, x_train, y_train, x_val=None, y_val=None):
        pass

    def predict(self, model, x_test):
        pass

    def get_model_out(self, x_train, y_train, x_test, n_fold=5):
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]

        train_oofp = np.zeros((n_train,))  # 存储每个fold的预测结果
        test_oofp = np.zeros((n_test, n_fold))  # 存储对测试集预测结果

        kfold = KFold(n_splits=n_fold, random_state=44, shuffle=True)

        for index, (ix_train, ix_val) in enumerate(kfold.split(x_train,y_train)):
            print ('{} fold of {} start train and predict...'.format(index, n_fold))
            X_fold_train = x_train[ix_train]
            y_fold_train = y_train[ix_train]

            X_fold_val = x_train[ix_val]
            y_fold_val = y_train[ix_val]

            model = self.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            #以4折做为训练数据训练的模型，预测剩下1折的验证数据，生成第一层的训练的输出数据
            train_oofp[ix_val] = self.predict(model, X_fold_val)
            
            #以4折生成的模型预测测试数据，生成第一层的测试集的输出数据
            test_oofp[:, index] = self.predict(model, x_test)
            
        #第一层的测试集输出数据
        test_oofp_mean = np.mean(test_oofp, axis=1)
        return train_oofp, test_oofp_mean

class GussianNBClassifier(StackingBaseClassifier):
    def __init__(self):
        # 参数设置
        pass

    def train(self, x_train, y_train, x_val, y_val):
        #print ('use GaussianNB train model...')
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        return gnb

    def predict(self, model, x_test):
        #print ('use GaussianNB model test... ')
        return model.predict(x_test)

class RFClassifer(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val, y_val):
        #print ('use RandomForest train model...')
        clf = RandomForestClassifier(n_estimators=25,max_depth=4,class_weight={0: 1,1: 4})
        clf.fit(x_train, y_train)
        return clf

    def predict(self, model, x_test):
        #print ('use RandomForest test...')
        return model.predict(x_test)

class LogisicClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        #print ('use LogisticRegression train model...')
        lr = LogisticRegression(class_weight={0: 1, 1: 4})
        lr.fit(x_train, y_train)
        return lr
    def predict(self, model, x_test):
        #print ('use LogisticRegression test...')
        return model.predict(x_test)

class DecisionClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        #print ('use DecisionClassifier train model...')
        dt = DecisionTreeClassifier(class_weight={0: 1, 1: 4},max_depth=5)
        dt.fit(x_train, y_train)
        return dt
    def predict(self, model, x_test):
        #print ('use DecisionClassifier test...')
        return model.predict(x_test)

def stacking_model_process(X_train,y_train,X_test,test_data_processed,outpath):
	#get_model_out:获取训练集的第一层输出，测试集的第一层输出

	gnb_cls = GussianNBClassifier()
	gnb_oop_train,gnb_oofp_val = gnb_cls.get_model_out(X_train,y_train,X_test)

	rf_cls = RFClassifer()
	rf_oop_train, rf_oofp_val = rf_cls.get_model_out(X_train, y_train, X_test)

	lg_cls = LogisicClassifier()
	lg_oop_train, lg_oofp_val = lg_cls.get_model_out(X_train, y_train, X_test)

	dt_cls = DecisionClassifier()
	dt_oop_train, dt_oofp_val = dt_cls.get_model_out(X_train, y_train, X_test)

	# 构造输入
	input_train = [gnb_oop_train,rf_oop_train,lg_oop_train,dt_oop_train]

	input_test = [gnb_oofp_val,rf_oofp_val,lg_oofp_val,dt_oofp_val]

	stacked_train = np.concatenate([data.reshape(-1,1) for data in input_train],axis=1)

	stacked_test = np.concatenate([data.reshape(-1,1) for data in input_test],axis=1)

	# stacking 第二层模型训练

	second_model = DecisionTreeClassifier(max_depth=3,class_weight={0: 1, 1: 4})
	second_model.fit(stacked_train,y_train)

	y_test_p = second_model.predict(stacked_test)

	test_index = np.array(test_data_processed["index"])
	
	print(len(test_index))
	
	with open(outpath, 'w') as fout:
		for index,pre in enumerate(y_test_p):
			if pre >=0.5:
				fout.write(str(test_index[index]) + '\t1\n')
			else:
				fout.write(str(test_index[index]) + '\t0\n')
if __name__ == '__main__':
	#读取数据
	w2c_train_all,train_all,test_data_df = read_all_train_data(train_data_path,train_add_data_path,sys.argv[1])
	#预处理词数据
	w2c_train_all_processed,train_all_processed,test_data_processed = preprocess_data(w2c_train_all,train_all,test_data_df)
	#预处理字数据
	w2c_train_all_char_processed = preprocessing_char(w2c_train_all)
	#训练Word2Vec词向量
	pre_train_w2v(w2c_train_all_processed,train_all_wordvec_path,binary = False)
	#处理后的语料训练词向量
	pre_train_char_w2v(w2c_train_all_char_processed,train_all_char_wordvec_path,binary = False)

	#提取训练数据和测试数据特征
	X ,X_test = extract_feature(train_all,test_data_df,train_all_processed,test_data_processed)
	#标签
	y = np.array(train_all_processed["label"].tolist())

	#模型训练预测
	stacking_model_process(X,y,X_test,test_data_processed,sys.argv[2])
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
