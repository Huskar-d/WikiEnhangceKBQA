import logging
import os
import traceback

import bs4
from bs4 import BeautifulSoup
import requests
import time
import codecs
import re
import json

from tqdm import tqdm

from tool import load_cache, persist_cache

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

requests.packages.urllib3.disable_warnings()

class WikiCrawler(object):
    def __init__(self):
        self.wiki_cache = {}
        self.topk = 2
        self.interval = 0
        self.dataset = 'WBQ'
        self.is_persist = True
        self.wiki_cache_path = 'data/wiki_info/wiki_cache.json'
        self.is_updated = False
        os.environ['http_proxy'] = "http://127.0.0.1:7890"
        self.wiki_cache = load_cache(self.wiki_cache_path)

    def crawl_all_path_wiki(self, interval, target_paths, use_cache=True, topk=3, is_persist=True):
        for path in tqdm(target_paths):
            if use_cache and path in self.wiki_cache:
                continue
            docs = self.crawl_wiki_content(path, topk)
            if docs is None:
                logger.error("%s crawl failed!!!!")
                continue
            time.sleep(interval)

    def crawl_wiki_content(self, keyword, topk=3, max_retry=3, use_cache = True):
        if keyword in self.wiki_cache and use_cache:
            return self.wiki_cache[keyword]
        logger.info('do crawl from wikipedia.org')
        retry_cnt = 0
        while retry_cnt < max_retry:
            try:
                base_url = 'https://en.wikipedia.org/wiki/'
                rlt = self.extract(base_url,keyword,topk)
                self.wiki_cache[keyword] = rlt
                self.is_updated = True
                logger.debug("craw success :%s"%keyword)
                if len(self.wiki_cache) %20 ==0:
                    persist_cache('data/%s/wiki_cache.json' % self.dataset, self.wiki_cache)
                    logger.info('persist wiki cache, current size:%s' % len(self.wiki_cache))
                return rlt
            except Exception as e:
                exstr = traceback.format_exc()
                logger.error(exstr)
                retry_cnt += 1
                logger.info('retrying(%s time) to craw topic :%s' % (retry_cnt, keyword))
                time.sleep(1)
        return None

    def extract_single_topic(self, doc_main_div):
        fetch_rlts = doc_main_div.find_all('p')

        paras = []

        for para in fetch_rlts:
            p_item = para.text.strip()
            if p_item != '':
                p_item = re.sub(r'\[[0-9]*\]', '', p_item)
                paras.append(p_item)

        # refer_list = {}
        # cur_key = ''
        # if 'may refer to:' in paras[0]:
        #     for child in doc_main_div.children:
        #         if not isinstance(child, bs4.element.Tag):  # 排除非标签元素干扰
        #             continue
        #         print(child.name)
        #         if child.name == 'h2':
        #             if cur_key != '':
        #                 refer_list[cur_key] = cur_list
        #             cur_key = child.text
        #             cur_list = []
        #         elif child.name == 'ul':
        #             for e_refer in child:
        #                 if not isinstance(e_refer, bs4.element.Tag):  # 排除非标签元素干扰
        #                     continue


        return '\n'.join(paras)

    def extract(self, base_url, keyword, topk):
        property = {}
        property['name'] = keyword
        property['docs'] = []

        url = base_url + keyword.strip().replace(' ', '_').replace('[', '').replace(']', '')
        url = url.replace('&apos;', '%27')  # '
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
        proxies = {'https':'127.0.0.1:7890'}
        html = requests.get(url, timeout=15, verify=False, headers=headers, proxies=proxies).text  # proxies=proxies
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table', class_='infobox')
        div = soup.find('div', id='mw-content-text')  # id='mw-content-text'

        # try:
        main_div = div.find('div', class_='mw-parser-output')
        if main_div:
            # first paragraph
            doc_text = self.extract_single_topic(main_div)
            property['docs'].append(doc_text)



            # sign and symptoms
            # sign = ''
            # if main_div.find('span', id='Signs_and_symptoms'):
            #     sign_h2 = main_div.find('span', id='Signs_and_symptoms').parent
            #     next_h2 = sign_h2.find_next_sibling('h2')
            #
            #     p1 = sign_h2.find_next_siblings('p')
            #     p2 = next_h2.find_previous_siblings('p')
            #
            #     p = [x for x in p1 if x in set(p2)]
            #
            #     for pp in p:
            #         sign += ''.join(pp.getText().strip())
            #     property['signSymptoms'] = sign
            # else:
            #     property['signSymptoms'] = 'None'

        # 查找最相关的词条https://github.com/AndreiRegiani/wikipedia-crawler.git
        else:

            new_url = div.find('a', class_='external text').get('href')
            html_new = requests.get(new_url, timeout=15, verify=False, headers=headers, proxies=proxies).text
            soup_new = BeautifulSoup(html_new, 'lxml')

            alternatives = soup_new.find_all('li', class_='mw-search-result')

            if alternatives:
                for i, alternative in enumerate(alternatives):
                    alternative_url = 'https://en.wikipedia.org' + alternative.find('a').get('href')
                    # print('Alternative url:', alternative_url)
                    alternative_html = requests.get(alternative_url, timeout=15, verify=False, headers=headers,
                                                    proxies=proxies).text
                    alternative_soup = BeautifulSoup(alternative_html, 'lxml')
                    alternative_table = alternative_soup.find('table', class_='infobox')
                    alternative_div = alternative_soup.find('div', id='mw-content-text')
                    alternative_main_div = alternative_div.find('div', class_='mw-parser-output')
                    doc_text = self. extract_single_topic(alternative_main_div)
                    property['docs'].append(doc_text)
                    if i+1 == topk:
                        break
        return property['docs']

    def __del__(self):
        if self.is_persist and self.is_updated:
            logger.info('dumping wiki cache')
            persist_cache(self.wiki_cache_path, self.wiki_cache)
            logger.info('dump finished')

if __name__ == '__main__':
    en_base_url = 'https://en.wikipedia.org/wiki/'
    keywords = ['Period'] # your own keywords
    #
    # count = 0
    # fpath = 'output.txt'
    # fp = codecs.open(fpath, 'w', encoding='utf-8')
    #
    # for k in keywords:
    #
    #     count = count+1
    #
    #     print('{}/{}...'.format(count, len(keywords)))
    #
    #     property = extract(en_base_url, k,2) # 具体抽取部分
    #
    #     id_dict = {'id': count}
    #     dict_final = dict(id_dict, **property)
    #
    #     json.dump(dict_final, fp, ensure_ascii=False)
    #     fp.write('\n')
    #
    #     time.sleep(3)
    #
    # fp.close()
    dataset = 'WBQ'
    interval = 0
    topk = 2
    crawler = WikiCrawler()
    # target_paths = fetch_target_path(dataset)
    for keyword in keywords:
        docs = crawler.crawl_wiki_content(keyword,2)
        logger.info("keyowrds:%s ==> %s"%(keyword, len(docs)))