
"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time
import random


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import sacrebleu
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from nmt_utils import read_corpus, batch_iter
from nmt_vocab import Vocab, VocabEntry

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
        Params:
            model (NMT): NMT Model
            dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
            batch_size (int): batch size. Default = 32
        Return:
            @returns ppl (float): perplexity on dev sentences
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score_old(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
        Params:
            references (List[List[str]]): a list of gold-standard reference target sentences
            hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
        Return:
            bleu_score(float): corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis], tokenizer:str ='nltk') -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
        Params:
            references (List[List[str]]): a list of gold-standard reference target sentences
            hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
            tokenizer (str): Tokenizer used (nltk or sp). Default = nltk
        Return:
            (float): corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # Old: using nltk corpus_bleu with tokenized words
    '''bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])'''

    # Reformat the sentences to feed them to sacrebleu
    if tokenizer == 'nltk':
        reformat_refs = [' '.join(ref) for ref in references]
        reformat_hyps = [' '.join(hyp.value) for hyp in hypotheses]
    elif tokenizer == 'sp':
        reformat_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
        reformat_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]
    else:
        raise Exception(f'unrecognised tokenizer {tokenizer}. Should be nltk or sp')

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(reformat_hyps, [reformat_refs])

    return bleu.score


def load_data_train(train_src_path, train_tgt_path, dev_src_path, dev_tgt_path, create_vocab=False, vocab_path=None, 
                    vocab_size=50000, vocab_cutoff = 2, subset=1.0, random_subset=False):
    """ Load dataset used by the NMT model
        Params:
            train_src_path (str): Path to the source sentences for training
            train_tgt_path (str): Path to the target sentences for training
            dev_src_path (str): Path to the source sentences for dev
            dev_tgt_path (str): Path to the target sentences for dev
            create_vocab (bool): If True, vocab will be created otherwise loaded from vocab_path
            vocab_path (str): Path to the json file with Vocab
            vocab_size (int): Size of vocabulary for both source and target languages. Default = 50000
            vocab_cutoff (int): if word occurs n < freq_cutoff times, drop the word. Default = 2
            subset (float): Percentage to apply to the train and dev sets in order to load a subset of the data. 
                Subset is a number > 0 and <= 1. Default = 1
            random_subset (bool): if True the data subset is random otherwise the first requierd elements of the data.
                Default = False  
        Return:
            train_data (list of (src_sent, tgt_sent)): list of tuples containing source and target 
                sentences for training
            dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target 
                sentences for dev
            vocab (Vocab): Vocab object for source and target
    """
    # read all data
    train_data_src = read_corpus(train_src_path, corpus_type='src')
    train_data_tgt = read_corpus(train_tgt_path, corpus_type='tgt')
    dev_data_src = read_corpus(dev_src_path, corpus_type='src')
    dev_data_tgt = read_corpus(dev_tgt_path, corpus_type='tgt')

    if subset == 1:
        train_data = list(zip(train_data_src, train_data_tgt))
        dev_data = list(zip(dev_data_src, dev_data_tgt))

    elif subset > 0 and subset < 1:
        num_train = int(subset*len(train_data_src))
        num_dev = int(subset*len(dev_data_src))
        if random_subset:
            train = list(zip(train_data_src, train_data_tgt))
            dev = list(zip(dev_data_src, dev_data_tgt))
            random.shuffle(train)
            random.shuffle(dev)
            train_data_src, train_data_tgt = zip(*train)
            dev_data_src, dev_data_tgt = zip(*dev)
        train_data = list(zip(train_data_src[:num_train], train_data_tgt[:num_train]))
        dev_data = list(zip(dev_data_src[:num_dev], dev_data_tgt[:num_dev]))

    else:
        raise ValueError(f'Incorrect value [{subset}] for subset; should be 0 < subset <=1')

    if create_vocab:
        src_sents, tgt_sents = zip(*train_data)
        vocab = Vocab.build(src_sents, tgt_sents, vocab_size, vocab_cutoff) 
    else:
        if vocab_path is not None:
            vocab = Vocab.load(vocab_path)
        else:
            raise ValueError(f'Vocab path is None and create_vocab is False')

    return train_data, dev_data, vocab

def train(train_data, dev_data, vocab, embed_size=256, hidden_size=256, dropout_rate=0.2, 
          uniform_init=0.1, device='cpu', lr=0.001, batch_size=32, clip_grad=5.0, log_every=10,
          valid_niter=2000, save_path='model.bin', patience=5, lr_decay=0.5, max_trials=5,
          max_epochs=30):
    """ Train the NMT model
        Params:
            train_data (list of (src_sent, tgt_sent)): list of tuples containing source and target 
                sentences for training
            dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target 
                sentences for dev
            vocab (Vocab): Vocab object for source and target
            embed_size (int): Embedding dimensionality. Default = 256
            hidden_size (int): Dimensionality for hidden states. Default = 256
            dropout_rate (float): Dropout probability. Default: 0.2
            uniform_init (float): If > 0: uniformly initialize all parameters
            device (str): device to perform the calc on. Default = 'cpu'
            lr (float): learning rate. Default = 0.001
            batch_size (int): batch size. Default = 32
            clip_grad (float): used in gradient clipping. Default = 5.0
            log_every (int): number of iterations to print stats. Default = 10
            valid_niter (int): number of iterations to perform validation. Default = 2000
            save_path (str): path to save the best model. Default: 'model.bin' in current dir
            patience (int): number of iterations to decay learning rate. Default = 5
            lr_decay (float): learning rate decay. Default = 0.5
            max_trials (int): terminate training after how many trials. Default = 5
            max_epochs (int): max number of epochs. Default = 30
        Return:
    """
    # Create NMT model and put it in train mode
    model = NMT(embed_size, hidden_size, vocab, dropout_rate)    
    model.train()

    # Uniformely initialize model parameters if required
    if np.abs(uniform_init) > 0.:
        print(f'uniformly init parameters [-{uniform_init}, +{uniform_init}]', file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    # Create target vocab mask with 0 for 'padding' index and 1 otherwise
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    # Set model device
    device = torch.device(device)
    model = model.to(device)
    print(f'Using device: {device}', file=sys.stderr)

    # Choose optimizer    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initializations
    num_trial = 0
    train_iter = 0
    current_patience = 0
    cum_loss = 0
    report_loss = 0
    cum_tgt_words = 0
    report_tgt_words = 0
    cum_examples = 0
    report_examples = 0
    epoch = 0
    valid_num = 0
    hist_valid_scores = []
    train_time = time.time()
    begin_time = time.time()
    
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        # Iterate over the batches in the training data
        for src_sents, tgt_sents in batch_iter(train_data, batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            current_batch_size = len(src_sents)
            
            # Calculate loss and backpropagate
            example_losses = -model(src_sents, tgt_sents) # (current_batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / current_batch_size # average loss 
            loss.backward()

            # clip gradient and update parameters
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print(f'epoch {epoch}, iter {train_iter}, ' \
                      f'avg. loss {report_loss / report_examples:.2f}, '\
                      f'avg. ppl {math.exp(report_loss / report_tgt_words):.2f}, ' \
                      f'cum. examples {cum_examples}, ' \
                      f'speed {report_tgt_words / (time.time() - train_time):.2f} words/sec, ' \
                      f'time elapsed {(time.time() - begin_time):.2f} sec', file=sys.stderr)
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print(f'epoch {epoch}, iter {train_iter}, cum. loss {cum_loss / cum_examples:.2f}, '\
                      f'cum. ppl {np.exp(cum_loss / cum_tgt_words):.2f} cum. examples {cum_examples}',
                      file=sys.stderr)
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1
                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print(f'validation: iter {train_iter}, dev. ppl {dev_ppl}', file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    # save model and optimizer state
                    print(f'save the best model to [{save_path}]', file=sys.stderr)
                    model.save(save_path)
                    torch.save(optimizer.state_dict(), save_path + '.optim') 
                    current_patience = 0
                    
                elif current_patience < patience:
                    current_patience += 1
                    print(f'hit patience {current_patience}', file=sys.stderr)

                    if current_patience == patience:
                        num_trial += 1
                        print(f'hit #{num_trial} trial', file=sys.stderr)
                        if num_trial == max_trials:
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        print(f'load previously best model and decay learning rate to {lr}', 
                              file=sys.stderr)

                        # load model
                        params = torch.load(save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        current_patience = 0

                if epoch == max_epochs:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)
    

def trainOld(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], corpus_type='src')
    train_data_tgt = read_corpus(args['--train-tgt'], corpus_type='tgt')

    dev_data_src = read_corpus(args['--dev-src'], corpus_type='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], corpus_type='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(test_src_path, test_tgt_path=None, model_path='model.bin', beam_size=5, max_decoding=70, device='cpu', output_path='output.txt'):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
        If the target gold-standard sentences are given, the function also computes
        corpus-level BLEU score.
        Params:
            test_src_path (str): Path to the test source file
            test_tgt_path (str): Path to the test target file (optional). Default=None
            model_path (str): Path to the model file generated after training. Default='model.bin'
            beam_size (int): beam size (# of hypotheses to hold for a translation at every step)
            max_decoding (int): maximum sentence length that Beam search can produce. Default=70
            device (str): device to perform the calc on. Default = 'cpu'
            output_path (str): Path for the output file to write the results of the translation. Default='output.txt'
    """

    print(f'load test source sentences from [{test_src_path}]', file=sys.stderr)
    test_data_src = read_corpus(test_src_path, corpus_type='src')
    
    if test_tgt_path is not None:
        print(f'load test target sentences from [{test_tgt_path}]', file=sys.stderr)
        test_data_tgt = read_corpus(test_tgt_path, corpus_type='tgt')

    print(f'load model from {model_path}', file=sys.stderr)
    model = NMT.load(model_path)
    model = model.to(torch.device(device))

    hypotheses = beam_search(model, test_data_src, beam_size=beam_size, max_decoding_time_step=max_decoding)

    if test_tgt_path is not None:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(output_path, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
        Params:
            model (NMT): NMT Model
            test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
            beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
            max_decoding_time_step (int): maximum sentence length that Beam search can produce
        Return: 
            hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    #retrieve args
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), f'Pytorch {torch.__version__}, should be >= 1.0.0'

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    device = "cuda:0" if args['--cuda'] else "cpu"
    
    if args['train']:

        train_src_path = args['--train-src']
        train_tgt_path = args['--train-tgt']
        dev_src_path = args['--dev-src']
        dev_tgt_path = args['--dev-tgt']
        train_batch_size = int(args['--batch-size'])
        clip_grad = float(args['--clip-grad'])
        valid_niter = int(args['--valid-niter'])
        log_every = int(args['--log-every'])
        model_save_path = args['--save-to']
        vocab_path = args['--vocab']
        embed_size = int(args['--embed-size'])
        hidden_size = int(args['--hidden-size'])
        dropout_rate = float(args['--dropout'])
        uniform_init = float(args['--uniform-init'])
        lr = float(args['--lr'])
        patience = int(args['--patience'])
        max_trials = int(args['--max-num-trial'])
        lr_decay = float(args['--lr-decay'])
        max_epochs = int(args['--max-epoch'])
        
        train_data, dev_data, vocab = load_data_train(train_src_path, train_tgt_path, dev_src_path, dev_tgt_path, create_vocab=True, 
                                            vocab_path=vocab_path, vocab_size=50000, vocab_cutoff = 2, subset=0.5, 
                                            random_subset=True)
        train(train_data, dev_data, vocab, embed_size=embed_size, hidden_size=hidden_size, dropout_rate=dropout_rate, 
              uniform_init=uniform_init, device=device, lr=lr, batch_size=train_batch_size, clip_grad=clip_grad, 
              log_every=log_every, valid_niter=valid_niter, save_path=model_save_path, patience=patience, 
              lr_decay=lr_decay, max_trials=max_trials, max_epochs=max_epochs)
    
    elif args['decode']:

        test_src_path = args['TEST_SOURCE_FILE']
        test_tgt_path = args['TEST_TARGET_FILE']
        model_path = args['MODEL_PATH']
        beam_size = int(args['--beam-size'])
        max_decoding = int(args['--max-decoding-time-step'])
        output_path = args['OUTPUT_FILE']

        decode(test_src_path=test_src_path, test_tgt_path=test_tgt_path, model_path=model_path, beam_size = beam_size,
               max_decoding=max_decoding, device=device, output_path=output_path)
    
    else:
        raise RuntimeError('invalid run mode')
    


if __name__ == '__main__':
    main()
