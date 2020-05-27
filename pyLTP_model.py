# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# create on 5/26/20
__author__ = 'sinsa'

import os
import logging
from logging import info, error, warn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PID:%(process)d - %(levelname)s: %(message)s')
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SentenceSplitter


class LTP_MODEL():
    def __init__(self):
        LTP_DATA_DIR = '/Users/yipu.si/Desktop/知识图谱/ltp_data_v3.4.0'  # ltp模型目录的路径
        info('loading models ...')
        self.segmentor = Segmentor()  # 初始化实例
        self.cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.segmentor.load(self.cws_model_path)  # 加载模型
        info('has loaded 分词模型')
        self.pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.postaggers = Postagger()  # 初始化实例
        self.postaggers.load(self.pos_model_path)  # 加载模型
        info('has loaded 词性标注模型')
        self.ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.recognizer = NamedEntityRecognizer()  # 初始化实例
        self.recognizer.load(self.ner_model_path)  # 加载模型
        info('has loaded 命名实体识别模型')
        self.par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        self.parser = Parser()  # 初始化实例
        self.parser.load(self.par_model_path)  # 加载模型
        info('has loaded 依存句法分析模型')

    def __release__(self):
        self.segmentor.release()  # 释放模型
        self.postaggers.release()  # 释放模型
        self.recognizer.release()  # 释放模型
        self.parser.release()  # 释放模型

    def SplitSentence(self, sentence):
        sents_list = SentenceSplitter.split(sentence)  # 分句
        return list(sents_list)

    def segment(self, input_list):
        '''
        功能：实现分词文本的分词
        返回值：每个文本的形成一个列表[['word1','word2'],['word1','word3'],……]
        '''
        segmented_text_list = []
        for text in input_list:
            words = self.segmentor.segment(text)  # 分词
            segmented_text_list.append(list(words))
        return segmented_text_list

    def postagger(self, input_list, return_words_list=False):
        '''
        功能：实现文本中每个词的词性标注
        返回值：每个文本是一个列表，列表中的每个词也是个列表[[['word1',u'O'],['word2',u'O']],[['word2',u'O'],['word5',u'O']],……]
        '''
        postagger_text_list = []
        words_list = self.segment(input_list)
        postags_list = []
        for words in words_list:
            postags = self.postaggers.postag(words)  # 词性标注
            postags_list.append(list(postags))
            words_postags = list(zip(words, list(postags)))
            postagger_text_list.append(words_postags)
        if return_words_list:
            return words_list, postags_list
        else:
            return postagger_text_list

    def NamedEntityRecognizer(self, input_list, Entity_dist=False, repead=False):
        '''
        功能：识别文本中的命名实体：地名，组织名和机构名
        参数repead：表示是否进行去重处理 ，默认是不去重
        参数Entity_dist：表示每个文本，返回的识别后的列表，还是抽取后的实体字典，默认返回的是列表
        返回值的形式：1.[[['word1',u'O'],['word2',u'O'],['word3',u'O']],[['word2',u'O'],['word3',u'O'],['word4',u'O']],……]
                      2.[{'person':[],'place':[],'organization':[]},{'person':[],'place':[],'organization':[]},{'person':[],'place':[],'organization':[]},……]
        '''

        words_list, postags_list = self.postagger(input_list, return_words_list=True)

        entity_text_list = []
        for words, postags in zip(words_list, postags_list):
            netags = self.recognizer.recognize(words, postags)  # 命名实体识别 人名（Nh）、地名（Ns）、机构名（Ni）
            text = list(zip(words, netags))
            entity_text_list.append(text)

        if Entity_dist:
            extract_entity_list = []
            for words_entity_note_list in entity_text_list:
                extract_entity_list.append(self.get_entity_dict(words_entity_note_list, repead))
            return extract_entity_list
        else:
            return entity_text_list

    def get_entity_dict(self, words_entity_note_list, repead):
        '''
        功能：根据实体识别的标志，统计文本中的命名实体
        参数repead：表示是否进行去重处理 ，默认是不去重
        返回值：{'person':[],'place':[],'organization':[]}
        '''
        '''
        O：这个词不是NE
        S：这个词单独构成一个NE
        B：这个词为一个NE的开始
        I：这个词为一个NE的中间
        E：这个词位一个NE的结尾
        Nh：人名
        Ni：机构名
        Ns：地名
        '''
        name_entity_dist = {}
        # 存储不同实体的列表
        name_entity_list = []
        place_entity_list = []
        organization_entity_list = []

        ntag_E_Nh = ""
        ntag_E_Ni = ""
        ntag_E_Ns = ""
        for word, ntag in words_entity_note_list:
            # print word+"/"+ntag,
            if ntag[0] != "O":
                if ntag[0] == "S":
                    if ntag[-2:] == "Nh":
                        name_entity_list.append(word)
                    elif ntag[-2:] == "Ni":
                        organization_entity_list.append(word)
                    else:
                        place_entity_list.append(word)
                elif ntag[0] == "B":
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word
                    else:
                        ntag_E_Ns = ntag_E_Ns + word
                elif ntag[0] == "I":
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word
                    else:
                        ntag_E_Ns = ntag_E_Ns + word
                else:
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word
                        name_entity_list.append(ntag_E_Nh)
                        ntag_E_Nh = ""
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word
                        organization_entity_list.append(ntag_E_Ni)
                        ntag_E_Ni = ""
                    else:
                        ntag_E_Ns = ntag_E_Ns + word
                        place_entity_list.append(ntag_E_Ns)
                        ntag_E_Ns = ""

        if repead:
            name_entity_dist['person'] = list(set(name_entity_list))
            name_entity_dist['organization'] = list(set(organization_entity_list))
            name_entity_dist['place'] = list(set(place_entity_list))
        else:
            name_entity_dist['person'] = name_entity_list
            name_entity_dist['organization'] = organization_entity_list
            name_entity_dist['place'] = place_entity_list
        return name_entity_dist

    def SyntaxParser(self, input_list, return_words_pos=False):
        '''
        # head = parent+1
        # relation = relate  可以从中间抽取head 和 relation 构成LTP 的标准输出，但是为了根据自己的情况，直接输出返回的全部的信息
        功能：实现依存句法分析
        返回值：每个文本的形成一个列表
        [[{u'relate': u'WP', u'cont': u'\uff0c', u'id': 4, u'parent': 3, u'pos': u'wp'},{u'relate': u'RAD', u'cont': u'\u7684', u'id': 1, u'parent': 0, u'pos': u'u'}],……]
        '''

        words_list, postags_list = self.postagger(input_list, return_words_list=True)

        syntaxparser_text_list = []
        for words, postags in zip(words_list, postags_list):
            arcs = self.parser.parse(words, postags)  # 句法分析
            res = [(arc.head, arc.relation) for arc in arcs]
            text = []
            for i in range(len(words)):
                tt = {
                    'id': i
                    , 'cont': words[i]
                    , 'pos': postags[i]
                    , 'parent': res[i][0]
                    , 'relate': res[i][1]
                }
                text.append(tt)
            syntaxparser_text_list.append(text)

        if return_words_pos:
            return words_list, postags_list, syntaxparser_text_list
        else:
            return syntaxparser_text_list

    def triple_extract(self, intput_list):
        '''
        功能: 对于给定的句子进行事实三元组抽取
        Args:
            sentence: 要处理的语句
                      形式是：'真实的句子'
        '''
        Subjective_guest = []  # 主谓宾关系(e1,r,e2)
        Dynamic_relation = []  # 动宾关系
        Guest = []  # 介宾关系
        Name_entity_relation = []  # 命名实体之间的关系
        # 分词后词的列表 words，词性列表 postags，实体标志列表 netags，语法分析列表 arcs
        words = []
        postags = []
        netags = []
        arcs = []
        syntaxparser_text_list = self.SyntaxParser(intput_list)
        entity_list = self.NamedEntityRecognizer(intput_list)
        for words_property_list in syntaxparser_text_list[0]:
            words.append(words_property_list['cont'])
            postags.append(words_property_list['pos'])
            arcs.append({'head': words_property_list['parent'], 'relation': words_property_list['relate']})
        for words_entity_list in entity_list[0]:
            netags.append(words_entity_list[1])

        child_dict_list = self.build_parse_child_dict(words, postags, arcs)

        for index in range(len(postags)):

            # 抽取以谓词为中心的事实三元组
            if postags[index] == 'v':
                child_dict = child_dict_list[index]
                # 主谓宾
                if 'SBV' in child_dict and 'VOB' in child_dict:
                    e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    r = words[index]
                    e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    Subjective_guest.append((e1, r, e2))

                # 定语后置，动宾关系
                if arcs[index]['relation'] == 'ATT':
                    if 'VOB' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, arcs[index]['head'] - 1)
                        r = words[index]
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        temp_string = r + e2
                        if temp_string == e1[:len(temp_string)]:
                            e1 = e1[len(temp_string):]
                        if temp_string not in e1:
                            Dynamic_relation.append((e1, r, e2))

                # 含有介宾关系的主谓动补关系
                if 'SBV' in child_dict and 'CMP' in child_dict:
                    # e1 = words[child_dict['SBV'][0]]
                    e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    cmp_index = child_dict['CMP'][0]
                    r = words[index] + words[cmp_index]
                    if 'POB' in child_dict_list[cmp_index]:
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                        Guest.append((e1, r, e2))

            # 尝试抽取命名实体有关的三元组
            if netags[index][0] == 'S' or netags[index][0] == 'B':
                ni = index
                if netags[ni][0] == 'B':
                    while netags[ni][0] != 'E':
                        ni += 1
                    e1 = ''.join(words[index:ni + 1])
                else:
                    e1 = words[ni]
                # 上面是抽取实体，没有判断是什么类型的实体。。
                if arcs[ni]['relation'] == 'ATT' and postags[arcs[ni]['head'] - 1] == 'n' and netags[
                    arcs[ni]['head'] - 1] == 'O':
                    r = self.complete_e(words, postags, child_dict_list, arcs[ni]['head'] - 1)
                    if e1 in r:
                        r = r[(r.index(e1) + len(e1)):]
                    if arcs[arcs[ni]['head'] - 1]['relation'] == 'ATT' and netags[
                        arcs[arcs[ni]['head'] - 1]['head'] - 1] != 'O':
                        e2 = self.complete_e(words, postags, child_dict_list, arcs[arcs[ni]['head'] - 1]['head'] - 1)
                        mi = arcs[arcs[ni]['head'] - 1]['head'] - 1
                        li = mi
                        if netags[mi][0] == 'B':
                            while netags[mi][0] != 'E':
                                mi += 1
                            e = ''.join(words[li + 1:mi + 1])
                            e2 += e
                        if r in e2:
                            e2 = e2[(e2.index(r) + len(r)):]
                        if r + e2 in sentence:
                            Name_entity_relation.append((e1, r, e2))
        return Subjective_guest, Dynamic_relation, Guest, Name_entity_relation

    def build_parse_child_dict(self, words, postags, arcs):
        """
        功能：为句子中的每个词语维护一个保存句法依存儿子节点的字典
        Args:
            words: 分词列表
            postags: 词性列表
            arcs: 句法依存列表
        """
        child_dict_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index]['head'] == index + 1:
                    if arcs[arc_index]['relation'] in child_dict:
                        child_dict[arcs[arc_index]['relation']].append(arc_index)
                    else:
                        child_dict[arcs[arc_index]['relation']] = []
                        child_dict[arcs[arc_index]['relation']].append(arc_index)
            child_dict_list.append(child_dict)
        return child_dict_list

    def complete_e(self, words, postags, child_dict_list, word_index):
        """
        功能：完善识别的部分实体
        """
        child_dict = child_dict_list[word_index]
        prefix = ''

        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])

        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix + words[word_index] + postfix


if __name__ == '__main__':
    intput_list = ['中国自称为炎黄子孙、龙的传人']
    model = LTP_MODEL()
    input_sentence = "雅生活服务的物业管理服务。"
    print(model.SplitSentence(input_sentence))
    print(model.segment(intput_list))
    print(model.postagger(intput_list))
    print(model.NamedEntityRecognizer(intput_list, Entity_dist=True))
    print(model.NamedEntityRecognizer(intput_list))
    print(model.SyntaxParser(intput_list))

    Subjective_guest, Dynamic_relation, Guest, Name_entity_relation = model.triple_extract(intput_list)

    print('=' * 30)
    print(Subjective_guest, Dynamic_relation, Guest, Name_entity_relation)
    model.__release__()
