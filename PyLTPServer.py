# !/usr/bin/env python
# -*- coding:utf-8 -*-
# create on 5/26/20
__author__ = 'sinsa'

import logging
from logging import info, error, warn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - PID:%(process)d - %(levelname)s: %(message)s')


from bottle import post, request, run
from pyltp import Parser
from pyltp import Postagger

pos_model = '/Users/yipu.si/Desktop/知识图谱/ltp_data_v3.4.0/pos.model'
dparser_model = '/Users/yipu.si/Desktop/知识图谱/ltp_data_v3.4.0/parser.model'

pos_parser = Postagger()
pos_parser.load(pos_model)
assert pos_parser is not None

dependancy_parser = Parser()
dependancy_parser.load(dparser_model)
assert dependancy_parser is not None


@post('/pos/')
def get_postagger():
    words = request.json['words']
    words = [w.encode('utf-8') for w in words]
    print(words)
    return {'result': ' '.join(pos_parser.postag(words))}


@post('/dparser/')
def get_dependeancy_parser():
    words = request.json['words']
    words = [w.encode('utf-8') for w in words]
    tags = list(pos_parser.postag(words))
    arcs = dependancy_parser.parse(words, tags)
    arcs = [(a.head, a.relation) for a in list(arcs)]

    return {
        'result': arcs
    }


run(host='0.0.0.0', port=3334, reloader=True)