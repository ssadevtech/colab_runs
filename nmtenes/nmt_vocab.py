"""
Usage:
    vocab.py --tok =<tokenizer> --src=<file> --tgt=<file> --f=<file> [options]

Options:
    -h --help                  Show this screen.
    --tok=<tokenizer>          Tokenizer [spm or nltk] 
    --src=<file>               File of training source sentences
    --tgt=<file>               File of training target sentences
    --f=<file>                 JSON file to dump the Vocab
    --size=<int>               Vocab size [default: 50000]
    --cutoff=<int>             Frequency cutoff to drop a word from the vocab [default: 2]
    --src-size=<int>           Vocab size for source language (with spm tokenizer)
    --tgt-size=<int>           Vocab size for target language (with spm tokenizer)
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from nmt_utils import read_corpus, pad_sents
import sentencepiece as spm

class VocabEntry(object):
    """ Vocabulary Entry: structure containing either src or tgt language terms
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
            Params:
                word2id (dict): dictionary mapping words to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1     # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the '<unk>' token if the word 
            is out of vocabulary.
            Params:
                word (str): word to look up.
            Return:
            index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry
            Params:
                word (str): word to look up
            Return:
                (bool): whether word is in VocabEntry    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
            Return
                (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used when printing the object.
        """
        return f'Vocabulary[size = {len(self)}]' 

    def id2word(self, wid):
        """ Return mapping of index to word
            Params:
                wid (int): word index
            Return:
                (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
            Params:
                word (str): word to add to VocabEntry
            Return:
            (int): index of the word
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
            into list or list of list of indices.
            Params:
                sents (list[str] or list[list[str]]): sentence(s) in words
            Return:
                (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
            Params:
                word_ids (list[int]): list of word ids
            Return:
                (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
            shorter sentences.
            Params:
                sents (List[List[str]]): list of sentences (words)
                device: device on which to load the tesnor, i.e. CPU or GPU
            Return:
                sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a VocabEntry.
            Params:
                corpus (list[list[str]]): corpus of text 
                size (int): # of words in vocabulary
                freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
            Return:
                vocab_entry (VocabEntry): VocabEntry instance produced from corpus provided
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'#words: {len(word_freq)}, #words with freq >= {freq_cutoff}: {len(valid_words)}')
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        """ Given a subword_list construct a VocabEntry.
            Params:
                subword_list (list[str]): list of subwords 
            Return:
                vocab_entry (VocabEntry): VocabEntry instance produced subword list provided
        """
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry

class Vocab(object):
    """ Vocab encapsulating src and target VocabEntry.
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab
            Params:
                src_vocab (VocabEntry): VocabEntry for source language
                tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        """ Build Vocabulary from source and target corpus
            Params:
                src_sents (list[list[str]]): Source sentences 
                tgt_sents (list[list[str]]): Target sentences
                vocab_size (int): Size of vocabulary for both source and target languages
                freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
            Return:
                (Vocab): Vocab instance produced from corpus provided
        """
        assert len(src_sents) == len(tgt_sents), 'src and tgt sentences have different lengths'

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        return Vocab(src, tgt)
    
    @staticmethod
    def build_spm(src_sents, tgt_sents) -> 'Vocab':
        """ Build Vocabulary from source and target subwords
            Params:
                src_sents (list[str]): Source subwords provided by SentencePiece
                tgt_sents (list[str]): Target subwords provided by SentencePiece
            Return:
                (Vocab): Vocab instance produced from corpus provided
        """

        print('initialize source vocabulary ..')
        src = VocabEntry.from_subword_list(src_sents)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump. Save word2id dict for both src and tgt VocabEntry
            Params:
                file_path (str): file path to vocab file
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
            Params:
                file_path (str): file path to vocab file
            Return: 
                (Vocab): Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used when printing the object.
        """
        return f'Vocab(source {len(self.src)} words, target {len(self.tgt)} words)'

def get_vocab_list(file_path, corpus_type, vocab_size):
    """ Use SentencePiece to tokenize and acquire list of unique subwords.
        Params:
            file_path (str): path to the file containing the corpus
            corpus_type(str): "src" (source language) or "tgt" (target language)
            vocab_size (int): number of unique subwords in vocabulary when reading and tokenizing.
        Return:
            sp_list (List[str]): List of unique subwords 
    """
    # train the spm model 
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=corpus_type, vocab_size=vocab_size) 
    # create an instance; this saves .model and .vocab files 
    sp = spm.SentencePieceProcessor() 
    # loads tgt.model or src.model                                                              
    sp.load('{}.model'.format(corpus_type))                                                              
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]
    return sp_list 

if __name__ == '__main__':
    args = docopt(__doc__)

    if args["--tok"] == 'nltk':
        print(f'read in source sentences: {args["--src"]}...')
        src_sents = read_corpus(args['--src'], corpus_type='src')
        
        print(f'read in target sentences: {args["--tgt"]}...')
        tgt_sents = read_corpus(args['--tgt'], corpus_type='tgt')

        print('generating vocab...')
        vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--cutoff']))
        print(f'Vocab generated: {vocab}')
        
        jfile = f'{args["--f"]}_{args["--size"]}.json' 
    
    elif args["--tok"] == 'spm':
        print(f'read in source sentences: {args["--src"]}...')
        src_sents = get_vocab_list(args['--src'], corpus_type='src', vocab_size=args["--src-size"])  
        
        print(f'read in target sentences: {args["--tgt"]}...')
        tgt_sents = get_vocab_list(args['--tgt'], corpus_type='tgt', vocab_size=args["--tgt-size"])

        print('generating vocab...')
        vocab = Vocab.build_spm(src_sents, tgt_sents)
        print(f'Vocab generated: {vocab}')

        jfile = f'{args["--f"]}_{args["--src-size"]}_{args["--tgt-size"]}.json' 
    else:
        raise NotImplementedError(f'Tokenizer: {args["--tok"]} not implemented')    

    
    vocab.save(jfile)
    print(f'vocabulary saved to {jfile}')