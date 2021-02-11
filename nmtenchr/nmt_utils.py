import math

import numpy as np
from tqdm import tqdm
import nltk
import sentencepiece as spm
nltk.download('punkt') # Only needed once

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings are at the end of each sentence.
        Params 
            sents (list[list[str]]): list of sentences (each sentence is a list of words)
            pad_token (str): padding token
        Return: 
            sents_padded (list[list[str]]): list of padded sentences
    """
    sents_padded = list(sents)
    lengths = [len(s) for s in sents_padded]
    max_len = max(lengths)
    for s,l in zip(sents_padded, lengths):
        s.extend([pad_token]*(max_len - l))
    
    return sents_padded


def read_corpus(file_path, corpus_type):
    """ Read file with sentences, where each sentence is in a seperate line
        Tokenize each sentence using nltk.word_tokenize
        Params:
            file_path (str): path to the file containing the corpus
            corpus_type(str): "src" (source language) or "tgt" (target language)
        Return:
            data (list[list[str]]): list of tokenized sentences. Target sentences have also a 
                beginning and end of sentence tokens ('<s>' and '</s>') 
    """
    data = []
    num_lines = sum(1 for line in open(file_path,'r'))

    with open(file_path,'r') as f:
        for line in tqdm(f, total=num_lines):
            sent = nltk.word_tokenize(line)
            if corpus_type == 'tgt':
                sent = ['<s>'] + sent + ['</s>']
            data.append(sent)

    return data

def read_corpus_spm(file_path, corpus_type, model_path=None):
    """ Read file with sentences, where each sentence is in a seperate line
        Tokenize each sentence using sentencepiece tokenizer
        Params:
            file_path (str): path to the file containing the corpus
            corpus_type(str): "src" (source language) or "tgt" (target language)
            model_path (str): path to the pre-trained sentencepiece model file. Default: '{corpus_type}.model'
        Return:
            data (list[list[str]]): list of tokenized sentences. Target sentences have also a 
                beginning and end of sentence tokens ('<s>' and '</s>')
            Note:
                A pre-trained sentencepiece tokenizer model needs to be available before this function is called 
    """
    data = []
    if model_path is None: model_path = f'{corpus_type}.model'
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    num_lines = sum(1 for line in open(file_path,'r'))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in tqdm(f, total=num_lines):
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if corpus_type == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length 
        of source sentence (largest to smallest)
        Params:
            data (list of (src_sent, tgt_sent)): list of tuples containing source and target 
                sentences
            batch_size (int): batch size
            shuffle (bool): whether to randomly shuffle the dataset
        Yield:
            src_sents (list[list[str]]), tg_sents (list[list[str]])
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents