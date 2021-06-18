import json
import os
import pathlib
import re
import random
import time
import logging
import traceback
from multiprocessing import Pool

from tqdm import tqdm
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from unidecode import unidecode

from tokenization import Tokenizer
from tool import persist_cache, load_cache
import WikiExtend.crawl as crawl

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_target_path(dataset_name, is_persist = True, is_refetch = False):
    kb_cache = load_cache('data/%s/kb_cache.json'%dataset_name)
    target_path = set()
    if not is_refetch:
        target_path = set(load_cache('data/%s/extend_paths.txt'%dataset_name,line_based=True, is_json=False))
        return target_path
    for k,v in kb_cache.items():
        for path in v:
            edges = re.findall(r"([a-z_]+)\.([a-z_]+)\.([a-z_]+)", path, flags=re.I)
            if edges is None:
                logger.warning('path:%s not matching'%path)
                continue
            for edge in edges:
                domain,type,property = edge
                target_path.update([type,property,type+' '+property])
    if is_persist:
        persist_cache('data/%s/extend_paths.txt'%dataset_name,list(target_path), line_based=True, is_json=False)
    return target_path


def get_stem_tokens(passage):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    passage = passage.lower()
    passage_tokens = nltk.word_tokenize(passage)
    passage_tokens_pos = nltk.pos_tag(passage_tokens)
    lemm_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(p)) for w, p in passage_tokens_pos]
    stem_tokens = [stemmer.stem(e) for e in lemm_tokens]
    stem_passage = ' '.join(stem_tokens)
    return stem_passage

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class WikiExtender(object):
    def __init__(self, crawler, tokenizer, do_persist= True):
        self.info_cache_path = 'data/wiki_info/relate_wiki_cache.json'
        self.relate_wiki_cache = load_cache(self.info_cache_path)
        self.crawler = crawler
        self.tokenizer = tokenizer
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.process_pool = Pool(1)
        self.do_persist = do_persist

    def get_last_1_hop_edge_from_path(self, path, use_str_path = False):
        '''
        :param path: m.04wgh travel.travel_destination.tour_operators ?d1	?d1 travel.\\ntour_operator.travel_destinations ?e2
        :return: [(domain, type, property),(domain, type, property)]
        '''
        # ((\?[d|e][0-9])|(([ |\t][m|g]|^[m|g])\.[a-zA-Z0-9_]+))
        if not use_str_path:
            if len(path) > 0:
                if len(path[-1])>3:
                    logger.warning('long path format:%s'%str(path[-1]))
                s1,r1, o1 = path[-1][-3:]
                e1 = r1.split('.')[-3:]
                target_edges = [e1]
                if '?d' in s1 or '?d' in o1 and len(path)>1:
                    if len(path[-2]) != 3:
                        logger.warning('error path format:%s' % str(path[-2]))
                    s2, r2, o2 = path[-2][-3:]
                    e2 = r2.split('.')[-3:]
                    target_edges = [e2, e1] 
            return target_edges

        elements = re.split('((([a-z0-9_]{2,}\.){2,})[a-z0-9_]{3,})', path,flags=re.I)
        if elements is None or len(elements) < 5:
            logger.warning('path:%s not matching' % path)
            return None
        o1  = elements[-1]
        domain1, type1, property1  = elements[-4].split('.')[-3:]
        s1 = elements[-5]
        target_edges = [(domain1, type1, property1)]
        # if CVT node, we need the last two edges
        if '?d' in o1 or '?d' in s1:
            o2 = elements[-5]
            domain2, type2, property2  = elements[-8].split('.')[-3:]
            s2 = elements[-9]
            target_edges = [(domain2,type2,property2),(domain1,type1,property1)]
        return target_edges

    def get_evidence_simple(self,target_edges):
        evidences = []
        for i in range(len(target_edges)):
            domain,type, property = target_edges[i]
            docs = self.crawler.crawl_wiki_content(property, topk=2)
            score = 0
            # precisely hit wiki
            paras = []
            if docs is not None and len(docs) > 0:
                paras = docs[0].split('\n')
            first_para = ''
            for para in paras:
                if len(para.split()) > 7 or 'may refer to:' in para:
                    first_para = para
                    break
            first_para = re.sub('\[.*\]', '', '%s' % first_para)
            # len(docs) == 1 means precisely hit
            if docs is not None and len(docs) == 1 and 'may refer to:' not in first_para and first_para != '':
                evidence_str = ''
                evidence_len = 0
                for sentence in first_para.split('.'):
                    sen_len = len(sentence.split())
                    if evidence_len +  sen_len> 35 and evidence_len > 0:
                        break
                    evidence_str += sentence
                    evidence_len += sen_len
                score = 2
            else:
                evidence_str = ''
                tokens = self.tokenizer.tokenize(property.replace('_',' '))
                if len(tokens) > 1:
                    for key in tokens:
                        if '#' in key:
                            continue
                        docs = self.crawler.crawl_wiki_content(key, topk=2)
                        paras = []
                        if docs is not None and len(docs) > 0:
                            paras = docs[0].split('\n')
                        first_para = ''
                        for para in paras:
                            if len(para.split()) > 7 or 'may refer to:' in para:
                                first_para = para
                                break
                        first_para = re.sub('\[.*\]', '', '%s' % first_para)

                        if docs is not None and len(docs) == 1 and 'may refer to:' not in first_para and first_para != '':
                            evidence_len = 0
                            for sentence in first_para.split('.'):
                                sen_len = len(sentence.split())
                                if evidence_len + sen_len > 18 and evidence_len > 0:
                                    break
                                evidence_str += sentence
                                evidence_len += sen_len
                score = 1 if evidence_str != '' else 0
            evidences.insert(0,(evidence_str,score))
        if len(evidences) == 1:
            evidences.append(('',-1))
        return evidences


    def extend_relate_wiki(self, question, candidate_path, topic_entities, m2n_cache):
        '''
        :param question: string, question string like"where is jamarcus russell from"
        :param candidate_path: string, candidate answer path in freebase, demo:"m.02hxv8 sports.sports_facility.home_venue_for ?d1\t?d1 type.object.type ?e2"
        :param wiki_cache: map, wiki cache that already crawled, {path1:[wiki_page1,wiki_page2]}
        :return: related_wikis: [passage1,passage2,...passage3]
        '''
        # for mid in topic_entities:
        #     print(m2n_cache[mid], topic_entities[mid])


        # tokens = nltk.word_tokenize(question)
        # pos_tags = nltk.pos_tag(tokens)

        # get target mid
        # rlt = re.findall(r'(([ |\t][m|g]|^[m|g])\.[a-zA-Z0-9_]+)', candidate_path)
        # # rlt : [('m.04wgh', 'm'), (' m.03_9hm', ' m')]
        # target_mids = [e[0].strip() for e in rlt]
        # get target edges
        target_edges = self.get_last_1_hop_edge_from_path(candidate_path,use_str_path=False)
        # target_edges = self.get_last_1_hop_edge_from_path(candidate_path, use_str_path=True)
        if target_edges is None:
            return [('',-1),('',-1)]

        relate_cache_key = '=>'.join([str(target_edges)])
        logger.info('target edge:%s'%(relate_cache_key))
        #relate_cache_key = '=>'.join([question, str(target_edges)])
        if self.relate_wiki_cache is not None and relate_cache_key in self.relate_wiki_cache:
            logger.debug('hit key in relate cache:%s'%relate_cache_key)
            return self.relate_wiki_cache[relate_cache_key]

        logger.info('compute relate wiki from the begining for :| %s'%relate_cache_key)

        evidences = self.get_evidence_simple(target_edges)
        if self.relate_wiki_cache is not None:
            self.relate_wiki_cache[relate_cache_key] = evidences
            # persist to disk periodly
            if len(self.relate_wiki_cache) % 10 ==0 and self.do_persist:
                persist_cache(self.info_cache_path,self.relate_wiki_cache)
        return evidences

        topic_entity_names = []
        exclude_words = []
        for mid in topic_entities:
            # assert(mid in topic_entities)
            if mid not in topic_entities:
                continue
            entity_name = re.sub('\W', ' ', m2n_cache[mid].lower())
            entity_name = unidecode(entity_name)
            topic_entity_names.append(entity_name)
            max_com_str = self.findcom(entity_name, question).strip()
            rlt = re.match('.*%s(\w+).*' % max_com_str, question)  # match subword
            if rlt is not None:
                max_com_str += rlt.group(1)
            exclude_words += max_com_str.split()

            # auxiliary output info
            rlt = re.match(r'(.*)%s(.*)' % max_com_str, question)
            remain_q = ('' if len(rlt.group(1)) < 2 else rlt.group(1)) + ('' if len(rlt.group(2)) < 2 else rlt.group(2))
            # print(re.findall(entity_name, question, flags=re.I))
            # print('question     :%s' % question)
            # print('question_pos:%s' % pos_tags)
            # print('lemm_tokens:%s' % lemm_tokens)
            # print('stem_tokens:%s' % stem_tokens)
            # print('entity:%s' % entity_name)
            # print('max_com_str:%s' % max_com_str)
            # print('remain_question:%s' % remain_q)
        if question == 'where does the parana river flow':
            question = question

        if (len(topic_entity_names)) == 0:
            # in true dataset this will not happen
            logger.warning('gold topic entity not in entitylinking rlt，for q ：%s' % question)
            return

        # fetch target words that should be related to wiki content
        target_words = []
        raw_target_words = []
        # add question's key word to target_words

        # for word, pos in pos_tags:
        #     if word in exclude_words:
        #         continue
        #     if word in ['which', 'to', 'what', 'who', 'where', 'when', 'did', 'were', 'was', 'is', 'do', 'are', 'does',
        #                 'of', 'the', 'for', 'in', 'they', 'my', '\'s', 'has']:
        #         continue
        #     lem_word = self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        #     stem_word = self.stemmer.stem(lem_word)
        #     target_words.append(stem_word)
        #     raw_target_words.append(word)
        # print('target_words:', " ".join(target_words))
        # add path edges to target_words
        if len(target_edges) == 1:
            edge_element_set = set([target_edges[0][1],target_edges[0][2]])
        elif len(target_edges) == 2:
            edge_element_set = set([target_edges[0][1], target_edges[0][2],target_edges[1][1], target_edges[1][2]])
        else:
            logger.warning('parse last hop path error, nothing found for:%s'%candidate_path)
        tmp_words = set()
        for edge in edge_element_set:
            tmp_words.update(re.split(r'\W+|_', edge))
        target_words += list(tmp_words)
        logger.info('target_words_with_path:%s'% " ".join(target_words))
        # get related wiki passages
        candidate_wikipages = set()
        for edge in target_edges:
            domain, type, property = edge
            keys = [type, property, type + ' ' + property]
            for key in keys:
                docs = self.crawler.crawl_wiki_content(key, topk=2)
                if docs is None:
                    logger.error("%s crawl failed!!!!")
                    continue
                # if property's wiki result is not precised, drop it.
                if key == property and len(docs) > 1:
                    continue
                candidate_wikipages.update(docs)
        # get top k related passage [(passage, mean_tfidf),...]
        evidences = self.find_relate_wikis_by_target_words(target_words,raw_target_words, candidate_wikipages)

        if self.relate_wiki_cache is not None:
            self.relate_wiki_cache[relate_cache_key] = evidences
            # persist to disk periodly
            if len(self.relate_wiki_cache) % 10 ==0:
                persist_cache(self.info_cache_path,self.relate_wiki_cache)
        return evidences

    def find_relate_wikis_by_target_words(self, target_words,raw_target_words, candidate_wikipages):
        relate_wiki = []
        e_topk = 2 #evidence's tok number
        corpus = []
        raw_corpus = []
        re_str = '|'.join(['(%s)' % word for word in raw_target_words])
        for wiki_page in candidate_wikipages:
            for passage in wiki_page.split('\n'):
                if len(passage) > 20 and re.search(re_str, passage, flags=re.I) is not None:
                    raw_corpus.append(passage)
        if len(raw_corpus) == 0:
            return [('',0.0) for _ in range(e_topk)]
        corpus = self.process_pool.map(get_stem_tokens, raw_corpus)
                # passage_tokens = nltk.word_tokenize(passage)
                # passage_tokens_pos = nltk.pos_tag(passage_tokens)
                # lemm_tokens = [self.lemmatizer.lemmatize(w, self.get_wordnet_pos(p)) for w, p in passage_tokens_pos]
                # stem_tokens = [self.stemmer.stem(e) for e in lemm_tokens]
                # corpus.append(' '.join(stem_tokens))
        words_idx, tfidf = self.get_ifidf(corpus)
        target_idx = [words_idx[word] for word in target_words if word in words_idx]
        passage_target_score = []
        for i in range(len(corpus)):
            total_score = np.array([tfidf[i, j] for j in target_idx]).sum()
            # sum of hit word's tf-idf score
            passage_target_score.append(total_score)
        sort_idx = np.argsort(-np.array(passage_target_score))

        # get top k related passage [(passage, mean_tfidf),...]
        # relate_wiki += [(raw_corpus[sort_idx[i]], passage_target_score[sort_idx[i]]) for i in range(2)]
        topk = min(len(sort_idx),e_topk)
        for i in range(topk):
            if passage_target_score[sort_idx[i]] < 1e-10:
                relate_wiki.append(('',0.0))
            else:
                sentences = raw_corpus[sort_idx[i]].split('.')
                target_passage = ''
                for sentence in sentences:
                    re_str = '|'.join(['(%s)'%word for word in target_words])
                    if re.search(re_str, sentence, flags=re.I) is not None:
                        target_passage += re.sub('\[.*\]','','%s.'%sentence)
                        logger.info('re:%s, info:%s'%(re_str, sentence))
                    if len(target_passage.split()) > 64:
                        break
                relate_wiki.append((target_passage, passage_target_score[sort_idx[i]]))
        if len(relate_wiki) < e_topk:
            relate_wiki += [('',0.0) for _ in range(e_topk-len(relate_wiki))]
        return relate_wiki



    def get_ifidf(self, corpus):
        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        # 计算个词语出现的次数
        X = vectorizer.fit_transform(corpus)
        # 获取词袋中所有文本关键词
        words = vectorizer.get_feature_names()
        words_idx = {}
        for i, word in enumerate(words):
            words_idx[word] = i
        # 类调用
        transformer = TfidfTransformer()
        # 将词频矩阵X统计成TF-IDF值
        tfidf = transformer.fit_transform(X)
        return words_idx, tfidf


    def findcom(self, str1, str2):
        xmax = 0  # 记录最大的值,即最大的字串长度
        xindex = 0  # 记录最大值的索引位置

        matrix = []

        for y, yitem in enumerate(str2):
            matrix.append([])  # 每次对str2迭代,生成新的子列表保存比对结果
            for x, xitem in enumerate(str1):
                if xitem != yitem:
                    matrix[y].append(0)  # 矩阵比较中,元素不同,置矩阵元素为0
                else:
                    if x == 0 or y == 0:  # 对处于矩阵第一行,第一列的直接置1,防止索引超界
                        matrix[y].append(1)
                    else:
                        matrix[y].append(matrix[y - 1][x - 1] + 1)  # 左上角的值+1

                    if matrix[y][x] > xmax:  # 此轮中最大的字串长度超过记录值
                        xmax = matrix[y][x]
                        xindex = x  # 最大值的索引位置

        return str1[xindex + 1 - xmax:xindex + 1]  # xindex+1因为后开特性,xindex+1后需往前回溯3个位置

    # def __del__(self):
    #     persist_cache(self.info_cache_path, self.relate_wiki_cache)
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['process_pool']
        return self_dict

    def persist_rlt(self):
        persist_cache(self.info_cache_path, self.relate_wiki_cache)


if __name__ == '__main__':
    # dataset = 'WBQ'
    # interval = 4
    # topk = 3
    # # path = 'm.04wgh travel.travel_destinatiom.tour_operators ?e1	?e1 travel.tour_operator.travel_destinations ?e2	m.03_9hm sports.sports_team.location ?e2'
    # path = 'm.06815z food.dish.type_of_dish1 ?e1'
    # get_last_1_hop_edge_from_path(path)


    crawler = crawl.WikiCrawler()
    relate_wiki_cache_path = 'data/wiki_info/relate_wiki_cache_test.json'
    tokenizer = Tokenizer('data/WBQ/vocab.txt')
    wiki_extender = WikiExtender(crawler, tokenizer, do_persist=False)
    e = [('popstra', 'celebrity', 'dated'), ('popstra', 'dated', 'participant')]
    wiki_extender.get_evidence_simple(e)
    wiki_extender.extend_relate_wiki('where do the blackhawks play',
                                     'm.0jnlm sports.sports_team.arena_stadium ?e1 last ?e1 architecture.structure.opened 1994-08-18',
                                     None,None)

    question = 'who is the voice of eric cartman on south park'
    m2n_cache = load_cache('data/WBQ/m2n_cache.json')


    topic_entities = load_cache('data/test_WBQ/te.json', line_based=True)
    gold_gs = load_cache('data/test_WBQ/g.txt',line_based=True, is_json=False)

    # topic_entities = load_cache('data/train_WBQ/te.json', line_based=True)
    # gold_gs = load_cache('data/train_WBQ/g.txt',line_based=True, is_json=False)
    #
    # with open('data/train_WBQ/q.txt') as f:
    with open('data/test_WBQ/q.txt') as f:
        i = 0
        err_num = 0
        hit_cnt = 0
        target_cnt = 0
        for question in f:
            question.strip()
            gold_gs[i]
            topic_entities[i]
            m2n_cache
            logger.info('question:%s' % question)
            logger.info('path(%s):%s' % (i, gold_gs[i]))
            if i == 76:
                i = i
            evidences = wiki_extender.extend_relate_wiki(question.strip(), gold_gs[i], topic_entities[i], m2n_cache)
            i += 1
            if evidences is None:
                err_num += 1
                logger.warning('question get non evidences:%s'%question)
                continue
            for e,score in evidences:
                hit_cnt += 1 if score >= 1 else 0
                target_cnt += 1 if score != -1 else 0
                # logger.info('evidence:%s'%e[:50])
                # logger.info('score:%s'%score)
            if i > 50000:
                break
        logger.info('total:%s'%i)
        logger.info('hit_cnt:%s'%hit_cnt)
        logger.info('target_cnt:%s'%target_cnt)
