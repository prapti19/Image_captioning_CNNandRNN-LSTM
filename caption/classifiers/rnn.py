from builtins import range
from builtins import object
import numpy as np

from caption.layers import *
from caption.rnn_layers import *


class CaptioningRNN(object):
   

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
       
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
       
      
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        
        
        inital_hidden, cache_initial = affine_forward(features, W_proj, b_proj)
        embedded_captions, cache_word_embedding = word_embedding_forward(captions_in, W_embed)
        if self.cell_type == 'rnn':
            rnn_outputs, cache_rnn = rnn_forward(embedded_captions, inital_hidden, Wx, Wh, b)
            scores, cache_scores = temporal_affine_forward(rnn_outputs, W_vocab, b_vocab)
        else:
            lstm_outputs,cache_lstm=lstm_forward(embedded_captions, inital_hidden, Wx, Wh, b)
            scores, cache_scores = temporal_affine_forward(lstm_outputs, W_vocab, b_vocab)
 
        
        loss, dsoftmax = temporal_softmax_loss(scores, captions_out, mask)
        
        dx_affineback,dw_a_b,db_a_b=temporal_affine_backward(dsoftmax,cache_scores)
        grads['W_vocab']=dw_a_b
        grads['b_vocab']=db_a_b
        if self.cell_type == 'rnn':
            dx_rnn, dh0_rnn, dWx_rnn, dWh_rnn, db_rnn=rnn_backward(dx_affineback,cache_rnn)
            grads['Wx']=dWx_rnn
            grads['b']=db_rnn
            grads['Wh']=dWh_rnn
            word_back=word_embedding_backward(dx_rnn,cache_word_embedding)
            grads['W_embed'] =word_back
            dx,dw_affine,db_affine=affine_backward(dh0_rnn,cache_initial)
            grads['W_proj'] = dw_affine
            grads['b_proj'] = db_affine
            
        else:
            dx_lstm, dh0_lstm, dWx_lstm, dWh_lstm, db_lstm=lstm_backward(dx_affineback,cache_lstm)
            grads['Wx']=dWx_lstm
            grads['b']=db_lstm
            grads['Wh']=dWh_lstm
            
            word_back=word_embedding_backward(dx_lstm,cache_word_embedding)
            grads['W_embed'] =word_back
            dx,dw_affine,db_affine=affine_backward(dh0_lstm,cache_initial)
            grads['W_proj'] = dw_affine
            grads['b_proj'] = db_affine
        
        
        
        ############################################################################
        #                             END                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
       
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

       
        
        cur_hidden_state, cache_initial = affine_forward(features, W_proj, b_proj)
        if self.cell_type == 'lstm':
            cur_c = np.zeros_like(cur_hidden_state)
        word_embed, _ = word_embedding_forward(self._start, W_embed)
        for i in range(max_length):
            
            
            if self.cell_type == 'rnn':
                
                
                cur_hidden_state, _ = rnn_step_forward(word_embed, cur_hidden_state, Wx, Wh, b)
                    
            elif self.cell_type == 'lstm':
                
                 cur_hidden_state,cur_c, _ = lstm_step_forward(word_embed, cur_hidden_state,cur_c ,Wx, Wh, b)
                
            cur_scores, _ = affine_forward(cur_hidden_state, W_vocab, b_vocab)
            captions[:,i] = np.argmax(cur_scores, axis=1)
            word_embed, _ = word_embedding_forward(captions[:, i], W_embed)
        ############################################################################
        #                             END                            #
        ############################################################################
        return captions
