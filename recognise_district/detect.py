# coding=utf-8
"""Detect the level infomation of area region that is detailed
into district from a squence of words.
"""
import os
import sys
from collections import OrderedDict
from functools import reduce
import pkgutil
import logging
import numpy as np
from . import regionbook_pb2

logger_name = '.'.join(__name__.split('.')[:-1] + ['recog.detect'])
logger = logging.getLogger(logger_name)


REGION_LEVELS = ['district', 'city', 'province']


def _load_region_data(pb2_path):
    regionbook = regionbook_pb2.LevelRegion()
    dat = pkgutil.get_data(__package__, pb2_path)
    regionbook.ParseFromString(dat)
    region_vocab = set()
    id2fullname = {}
    id2names = {}
    region2id = {}
    id2position = {}
    for level_name in REGION_LEVELS:
        for position, addr in enumerate(
                            regionbook.__getattribute__(level_name).address):
            fullname = addr.fullname
            name = addr.name
            idx = int(addr.id)
            alias = addr.alias
            names = [fullname, name]
            if name == '':
                names = [fullname]
            names = set(names+list(alias))
            region_vocab.update(names)
            id2fullname[idx] = fullname
            id2names[idx] = list(names)
            for ele in names:
                region2id[ele] = region2id.get(ele, []) + [idx]
            id2position[idx] = position
    region_dat = {'region_vocab': region_vocab, 'region2id': region2id,
                  'id2fullname': id2fullname, 'id2names': id2names,
                  'id2position': id2position}
    return region_dat, regionbook


class DetectDistrict(object):
    """Detector which accept inputs and use extra data to detect.
    """

    def __init__(self, params, pb2_path):
        region_dat, pb2 = _load_region_data(pb2_path)
        self.pb2 = pb2
        self.region_dat = region_dat

        self.params = params
        self.data_inferred = None
        # 直辖市，港澳无二级编码
        self.non_exist_code = set([110100, 120100, 310100, 500100,
                                   500200, 810100, 820100])
        self.special_code = set([110000, 120000, 310000, 500000,
                                 810000, 820000])
        self.hits = {'id': set(), 'freq': None}
        self.paths = []
        self.path_overlap = []

    def __delay_load(self):
        import gensim
        d = os.path.dirname(sys.modules[__package__].__file__)
        wv = gensim.models.KeyedVectors.load_word2vec_format(
                os.path.join(d, 'data/loc.vec.bin'), binary=True)
        self.data_inferred = {'wv': wv, 'vocab': set(wv.vocab.keys())}
        logger.debug('inferred delay loading over: %s' % d)

    def __handle(self, words):
        hits = set(words) & self.region_dat['region_vocab']
        self.words = words
        hit_freq = OrderedDict()
        for w in words:
            if w not in hits:
                continue
            for idx in self.region_dat['region2id'][w]:
                if idx in hit_freq:
                    hit_freq[idx] += 1
                else:
                    hit_freq[idx] = 1

        self.hits['id'] = list(hit_freq.keys())
        self.hits['freq'] = hit_freq
        self.paths = []
        self.path_overlap = []
        logger.debug('input words: %s' % words)
        logger.debug('hit: %s' % hits)
        logger.debug('hit freq: %s' % self.hits['freq'])

    def _back_path(self):
        """ 回溯新闻中命中地域的路径
        """
        self.paths = []
        for id in self.hits['id']:
            subpath = [id]
            r = 100
            while r <= 10e4:
                pre_id = id
                id = id // r * r
                if pre_id % r != 0 and id not in self.non_exist_code:
                    subpath.append(id)
                r *= 100
            self.paths.append(subpath)
        logger.debug('back path: %s' % self.paths)

    def _get_overlap(self):
        """ 计算回溯区域路径上与新闻中地点的重合度
        """
        path_overlap = np.ones(len(self.paths), dtype='int')
        for idx, path in enumerate(self.paths):
            if len(path) > 1:
                path_overlap[idx] = len(set(self.hits['id']) & set(path))
        self.path_overlap = path_overlap
        logger.debug('path overlap: %s' % self.path_overlap)

    def _get_candidate_path(self):
        """ 得到候选路径
        """
        candidate_paths = []
        max_val = np.max(self.path_overlap)
        topn = np.where(self.path_overlap == max_val)[0]
        topn_paths = list(map(lambda x: self.paths[x], topn))
        logger.debug('topn paths: %s' % topn_paths)
        # 计算频率
        count, rk = self._count_freq(topn_paths, len(topn))
        logger.debug('count freq: %s' % count)
        # 新闻越长，要求地域信息越明确
        len_penalty = len(self.words) / self.params['penalty_factor']
        len_penalty = 1 if len_penalty < 1 else len_penalty
        # 只出现一个地域
        if len(topn) == 1:
            if max_val >= (self.params['limit'] * len_penalty):
                return [topn_paths[0]]
            else:
                return []
        # 频率判断
        if count[rk[0]] < (self.params['freq_threshold'] - 1) * len_penalty:
            logger.debug('freq end')
            return []

        if (count[rk[0]] > count[rk[1]] * self.params['alpha'] or
            count[rk[1]] < (self.params['freq_threshold'] *
                            len(self.words) / self.params['penalty_factor'])):
            candidate_paths.append(topn_paths[rk[0]])
        else:
            candidate_paths = list(map(lambda x: topn_paths[x], rk[:2]))

        # TODO: 路径长度=1，新闻细节判断
        logger.debug('candidate paths: %s' % candidate_paths)

        return candidate_paths

    def detect(self, words, confidence=True):
        self.__handle(words)
        # 地点出现频率小于阈值
        hit_count = sum([self.hits['freq'].get(ele)
                         for ele in self.hits['freq']])
        logger.debug('hit freq: %s' % hit_count)
        blank_rsp = []
        if confidence:
            blank_rsp = [-1] * 3

        # 地域词判断条件
        if hit_count < self.params['limit']:
            return blank_rsp
        self._back_path()
        if len(self.paths) < 1:
            return blank_rsp
        self._get_overlap()
        paths = self._get_candidate_path()
        if len(paths) < 1:
            return blank_rsp
        results = []
        for path in paths:
            if len(set(path) & self.special_code) > 0:
                path.append(path[-1])
            pad_val = -1
            pad_width = (3-len(path), 0)
            path_padded = np.pad(path, pad_width, 'constant',
                                constant_values=pad_val)
            path_padded = [idx if idx in self.region_dat['id2fullname']
                           else -1 for idx in path_padded]
            results.append(np.array(path_padded))
        if confidence:
            if self.data_inferred is None:
                self.__delay_load()
            samples = [path[np.where(path != -1)[0][0]]
                       for path in results]
            confidence_id = self.__most_similar(samples, [])
            result = results[np.where(samples == confidence_id)[0][0]]
            return result.tolist()
        else:
            return [path.tolist() for path in results]

    def detect_with_infer(self, words, step=1, local_mode=False):
        if self.data_inferred is None:
            self.__delay_load()

        path = np.array(self.detect(words))
        logger.debug('infer input path: %s' % path)
        step = min(step, 3)
        while step > 0:
            path = self.__inferring(path, local_mode)
            logger.debug('infer step %s path: %s' % (step, path))
            step -= 1
        return path

    def __get_TBD_regions(self, region_id, level):
        """ 获取待确定区域范围
        Args:
            region_id: 粒度最小一级的确定区域id
            level: 粒度最小一级行政区域
        """
        position = self.region_dat['id2position'].get(region_id, -1)
        if position < 0:
            return None
        if region_id in self.special_code:
            level += 1
        level_name = REGION_LEVELS[level]
        logger.debug('infer level: %s' % level_name)
        last_region = self.pb2.__getattribute__(level_name).address[position]
        logger.debug('infer last region: %s' % last_region)
        start = last_region.cidx.start
        end = last_region.cidx.end
        TBD_regions = []
        for addr in self.pb2.__getattribute__(
                REGION_LEVELS[level-1]).address[start: end+1]:
            TBD_regions.append(int(addr.id))
        return TBD_regions

    def __most_similar(self, TBD_regions, pre_region_names, local_mode=False):
        wv = self.data_inferred['wv']
        vocab = self.data_inferred['vocab']
        pre_region_names = set(pre_region_names)

        def get_embs(tokens):
            embs = np.zeros([len(tokens), wv.vector_size])
            for idx, w in enumerate(tokens):
                if w not in wv.vocab:
                    logger.debug('infer vocab %s not found' % w)
                    continue
                embs[idx] = wv.get_vector(w)
            emb = embs.mean(axis=0)
            return emb

        benchmark = set(self.words) - pre_region_names
        if local_mode:
            tokens = [self.data_inferred['related_token'].get(addr_id, set())
                      for addr_id in TBD_regions]
            benchmark = [ele & benchmark for ele in tokens]
            logger.debug('inferring benchmark: %s' % benchmark)
            benchmark = reduce(lambda x, y: x | y, benchmark)
        else:
            benchmark = benchmark & vocab
        logger.debug('benchmark words: %s' % list(benchmark)[:100])
        benchmark_emb = get_embs(benchmark)
        candidates_embs = np.zeros([len(TBD_regions), wv.vector_size])
        for idx, ws in enumerate(TBD_regions):
            TBD_name = self.region_dat['id2fullname'].get(ws, None)
            logger.debug('index: %s, candidate name: %s' % (idx, TBD_name))
            if TBD_name is None:
                TBD_name = []
            else:
                TBD_name = [TBD_name]
            candidates_embs[idx] = get_embs(TBD_name)
        cos_distance = 1 - np.dot(candidates_embs, benchmark_emb) / (
                            np.linalg.norm(candidates_embs, axis=1) *
                            np.linalg.norm(benchmark_emb) + 1e-5)
        logger.debug('infer cos distance: %s' % cos_distance)
        return TBD_regions[np.argmin(cos_distance)]

    def __inferring(self, path, local_mode):
        """根据已有信息进一步推断未知地域信息
        推断下一步

        Args:
            path: hitting path: [-1, 440300, 440000]
            short_text: True: use origin text as benchmark

        Returns:
            next step inferred path: [440305, 440300, 440000]
        """
        uncertain_level = np.where(path == -1)[0]
        if len(uncertain_level) < 1:
            return path
        last_level = uncertain_level[-1] + 1
        logger.debug('infer last level: %s' % last_level)

        if last_level > 2:
            TBD_regions = [int(addr.id) for addr in self.pb2.province.address]
        else:
            TBD_regions = self.__get_TBD_regions(path[last_level], last_level)
        logger.debug('TBD regions: %s' % TBD_regions)

        pre_region_names = []
        for i in range(3- last_level):
            pre_region_names.extend(
                    self.region_dat['id2names'].get(path[last_level-i], []))
        logger.debug('previous regions: %s' % pre_region_names)

        if TBD_regions is None:
            logger.info('TBD regions: %s not found.' % path[last_level])
            return path
        inferred_id = self.__most_similar(TBD_regions, pre_region_names,
                                          local_mode)
        path[uncertain_level[-1]] = inferred_id
        return path

    def _count_freq(self, paths, size):
        """ 计算词频
        """
        count = np.zeros(size, dtype='int')
        for i, path in enumerate(paths):
            freq = [self.hits['freq'].get(idx, 0) for idx in path]
            count[i] = np.sum(freq)
        rk = np.argsort(-count)
        return count, rk

    def _count_w2v(self, paths):
        """Abandon
        计算词向量相近的ner
        """
        count = np.zeros(len(paths), dtype='int')
        res = []
        for idx, path in enumerate(paths):
            fullname = self.region_dat['id2fullname'][path[0]]
            similar_words = []
            try:
                similar_words = self.word_vectors.similar_by_word(
                                fullname, topn=50)
            except:
                continue
            logger.debug(fullname + ' similar word: %s' % similar_words[:5])
            similar_words = set([ele[0] for ele in similar_words])
            shared_words = (similar_words & set(self.words)
                            - self.region_dat['region_vocab'])
            logger.debug('shared word: %s' % shared_words)
            count[idx] = len(shared_words)
        rk = np.argsort(-count)
        if count[rk[0]] > count[rk[1]] * 1.2:
            res.append(paths[rk[0]])
        else:
            res = paths[:2]
        return res

    def path2name(self, path):
        nodes = [self.region_dat['id2fullname'][node]
                 if node in self.region_dat['id2fullname'] else ''
                 for node in path]
        return nodes

    def path2str(self, paths):
        names = self.path2name(paths)
        if len(names) < 1:
            return ''
        return ' '.join(names)

    def detect_for_fullname(self, words, confidence=True):
        if confidence:
            path = self.detect(words)
            return self.path2name(path)
        else:
            return [self.path2name(path) for path in self.detect(words, confidence)]
