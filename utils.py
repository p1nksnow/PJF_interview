from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import jieba
import argparse
import re
from text2vec import Similarity
from text2vec.similarity import SimilarityType,EmbeddingType
import json
from sklearn import metrics
import numpy as np

def load_split_dataset(doc_path,sample_path):
    with open(doc_path,'r') as t:
        doc_data = json.load(t)
    with open(sample_path,'r') as t:
        samples = json.load(t)
    split_ratio = 0.6  # 0.6
    train_sample = {'geek':{},'boss':{}}
    test_sample = {'geek':{},'boss':{}}
    
    for gid,sample in list(samples['geek'].items()):
        train_sample['geek'][gid] = {'pos':[],'neg':[]}
        test_sample['geek'][gid] = {'pos':[],'neg':[]}
        
        train_sample['geek'][gid]['pos'] = sample['pos'][:int(split_ratio*len(sample['pos']))]
        train_sample['geek'][gid]['neg'] = sample['neg'][:int(split_ratio*len(sample['neg']))]
        test_sample['geek'][gid]['pos'] = sample['pos'][int(split_ratio*len(sample['pos'])):]
        test_sample['geek'][gid]['neg'] = sample['neg'][int(split_ratio*len(sample['neg'])):]
    
    for bid,sample in list(samples['boss'].items()):
        train_sample['boss'][bid] = {'pos':[],'neg':[]}
        test_sample['boss'][bid] = {'pos':[],'neg':[]}
        
        train_sample['boss'][bid]['pos'] = sample['pos'][:int(split_ratio*len(sample['pos']))]
        train_sample['boss'][bid]['neg'] = sample['neg'][:int(split_ratio*len(sample['neg']))]
        test_sample['boss'][bid]['pos'] = sample['pos'][int(split_ratio*len(sample['pos'])):]
        test_sample['boss'][bid]['neg'] = sample['neg'][int(split_ratio*len(sample['neg'])):]
    

    return doc_data,train_sample,test_sample
