import sys
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class ModelEmbeddings(nn.Module): 
    """ Class that converts input words to their embeddings
    """
    def __init__(self, embed_size, vocab):
        """ Init the Embedding layers
            Params:
                embed_size (int): Embedding dimensionality
                vocab (Vocab): Vocab object containing src and tgt languages (see nmt_vocab.py)
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        self.source = nn.Embedding(len(vocab.src), embed_size, padding_idx=src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=tgt_pad_token_idx)

class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
        - See pdf of assignment 4 (cs224n) for details
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model
            Params:
                embed_size (int): Embedding dimensionality (emb)
                hidden_size (int): Dimensionality of hidden states (h)
                vocab (Vocab): Vocab object containing src and tgt languages (see nmt_vocab.py)
                dropout_rate (float): Dropout probability
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0

        # Encoder: Bidirectional LSTM with bias
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, 
                               bias=True, batch_first=False, bidirectional=True)
        # Decoder: LSTM cell with bias
        self.decoder = nn.LSTMCell(input_size=hidden_size+embed_size, hidden_size=hidden_size, 
                                   bias=True)
        # Linear projection of the encoder final hidden state (concatenation of both forward and 
        # backward) to initialize decoder first hidden state. No bias
        self.h_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, 
                                      bias=False)
        # Linear projection of the encoder final cell state (concatenation of both forward and 
        # backward) to initialize decoder first cell state. No bias
        self.c_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, 
                                      bias=False)        
        # Linear projection to calculate multiplicative attention. No bias
        self.att_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, 
                                        bias=False) 
        # Linear projection applied to the concatenation of the attention output and the decoder
        # hidden state. Used in the calc of the combined-output vector. No bias
        self.combined_output_projection = nn.Linear(in_features=3*hidden_size, 
                                                    out_features=hidden_size, bias=False) 
        # Linear projection applied to the combined-output vector before the result is passed to
        # a softmax to generate a distribution over the target vocab
        self.target_vocab_projection = nn.Linear(in_features=hidden_size, 
                                                 out_features=len(vocab.tgt), bias=False) 
        # Dropout layer. Used in the calculation of the combined-output vector
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
            target sentences under the language models learned by the NMT system.
            Params:
                source (List[List[str]]): list of source sentence tokens
                target (List[List[str]]): list of target sentence tokens, wrapped by 
                    `<s>` and `</s>`
            Returns:
                scores (Tensor of shape (b, )): Represent the log-likelihood of generating 
                the gold-standard target sentence for each example in the input batch. 
                b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device) #(l_src, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device) #(l_tgt, b)

        # Run the network forward:
        #  1. Apply the encoder to `source_padded` by calling `self.encode()`
        #  2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        #  3. Apply the decoder to compute combined-output by calling `self.decode()`
        #  4. Compute log probability distribution over the target vocabulary using the
        #     combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded) #(l_tgt, b, h)
        proba = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1) #(l_tgt, b, vocab_tgt)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float() #(l_tgt, b)
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(proba, index=target_padded[1:].unsqueeze(-1), 
                                                  dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, 
              source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain 
                initial states for the decoder.
            Params:
                source_padded (Tensor with shape (l_src, b)): Tensor of padded source sentences 
                sorted from longest to shortest. Contains the indices of the tokens. l_src = 
                max src sentence length and b = batch size
                source_lengths (List[int]): List of actual lengths for each of the source sentences 
                    in the batch
            Return:
                enc_hiddens (Tensor with shape (b, l_src, 2*h)): Tensor of hidden states for 
                    each step in the sequence. h = hidden size
                dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's
                    initial hidden state and cell state. Each tensor is of shape (b, 2*h)
        """
        # Retrieve embeddings of source sentences -> shape (seq_len, batch, emb_size)
        x = self.model_embeddings.source(source_padded)
        
        # Pack padded input
        x_packed = pack_padded_sequence(input=x, lengths=source_lengths, batch_first=False, 
                                        enforce_sorted= True)
        # Retrieve the outputs of the LSTM
        output, (last_hidden, last_cell) = self.encoder(x_packed)
        
        # Pad packed hidden states and put batch first -> shape (batch, seq_len, 2*hidden_size)
        enc_hiddens, _ = pad_packed_sequence(output, batch_first=True)
        
        # Compute initial hidden and cell states for the decoder by concatenating the forward
        # and backward states --> shape (batch, 2*hidden_size)
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], 
               target_padded: torch.Tensor) -> torch.Tensor:
        """ Compute the combined-output vectors for a batch.
            Params:
                enc_hiddens (Tensor with shape (b, l_src, 2*h)): Tensor of hidden states 
                    for each step in the sequence. l_src = max src sentence length, 
                    b = batch size, h = hidden size
                enc_masks (Tensor with shape (b, l_src)): Contains 1 in positions correponding to
                'pad'tokens in the input and 0 otherwise
                dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's
                    initial hidden state and cell state. Each tensor is of shape (b, 2*h)
                target_padded (Tensor with shape (l_tgt, ba)): Padded 
                    target sentenes. l_tgt = max target sentence length
            Return:
                combined_outputs (Tensor with shape (l_tgt, b, h)): combined-output vectors
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined-output vector (o{t-1}) as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # List used to collect the combined-output vectors (o{t}) on each step
        combined_outputs = []

        # Retrieve embeddings of target sentences -> shape (seq_len, batch, emb_size)
        y = self.model_embeddings.target(target_padded)
        
        # Apply attention projection layers to encoder hidden states
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        
        # For each token in the target sequence:
        #   Squeeze y_t from (1, batch, emb) to (batch, emb)
        #   Concatenate y_t with previous combined-output vectors
        #   Perform decoder step to compute next hidden and cell states and new combined-outputs
        for y_t in torch.split(y, 1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)
            ybar_t = torch.cat((y_t, o_prev), dim=1)
            dec_state, o_t, _ = self.step(ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, 
                                          enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        # conversion from list length tgt_len of tensors shape (b, h), to a single tensor 
        # shape (l_tgt, b, h)
        combined_outputs = torch.stack(combined_outputs, dim=0)

        return combined_outputs


    def step(self, ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
            Params:
                ybar_t (Tensor with shape (b, emb + h)): Decoder input: Concatenated Tensor of [
                    y_t o_prev]. b = batch size, emb = embedding size and h = hidden size
                dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h)
                    previous hidden state (0) and previous cell state (1)
                    
                enc_hiddens (Tensor with shape (b, l_src, 2*h)): Tensor of hidden states 
                    for each step in the sequence. l_src = max src sentence length
                enc_hiddens_proj (Tensor with shape (b, src_len, h)): Encoder hidden states Tensor
                    projected from (2*h) to h. 
                enc_masks (Tensor with shape (b, l_src)): Contains 1 in positions correponding to
                'pad'tokens in the input and 0 otherwise
            Returns:
                dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h)
                    decoder's new hidden state (0) and decoder's new cell state (1)
                combined_output (Tensor with shape (b, h)): Combined output Tensor at timestep t
                    e_t (Tensor with shape (b, src_len)): attention scores distribution
        """

        combined_output = None

        # New decoder hidden and cell states
        dec_state = self.decoder(ybar_t, dec_state)
        dec_hidden, _ = dec_state # cell state not needed below
        
        # Attention scores - shape (b, l_src)
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)

        # Set e_t to -inf where enc_masks has 1
        # i.e. set attention scores with 'pad' tokens to -inf 
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
        
        # Distribution of attention scores
        alpha_t = F.softmax(e_t, dim=1)
        
        # Weighted sum of encoder hidden states based on attention distribution
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens), dim=1)
        
        # Compute the combined-output vectors
        u_t = torch.cat((a_t, dec_hidden), dim=1)
        v_t = self.combined_output_projection(u_t)
        o_t = self.dropout(torch.tanh(v_t))

        combined_output = o_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
            Params
                enc_hiddens (Tensor with shape (b, l_src, 2*h)): Tensor of hidden states 
                    for each step in the sequence. l_src = max src sentence length, 
                    b = batch size, h = hidden size
                source_lengths (List[int]): List of actual lengths for each of the sentences 
                    in the batch.
            Return:
                enc_masks (Tensor with shape (b, l_src)): Contains 1 in positions correponding to
                'pad'tokens in the input and 0 otherwise
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
            Params:
                src_sent (List[str]): a single source sentence (words)
                beam_size (int): beam size
                max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
            Return:
                hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                    value: List[str]: the decoded target sentence, represented as a list of words
                    score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            #prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt) #changed / to // due to runtime error
            prev_hyp_ids = torch.floor_divide(top_cand_hyp_pos, len(self.vocab.tgt))
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)