import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size=512, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.word_embedding = nn.Embedding(self.embed_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.to_out_size = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions):
        '''
            features: shape -> (batch_size, embed_size)

            output: shape -> (batch_size, caption_length, vocab_size)
        '''
        # Remove the <end> word
        captions = captions[:, :-1]
        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size, self.device)
        embeddings = self.word_embedding(captions)
        output = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        output, self.hidden = self.gru(output, self.hidden)
        output = self.to_out_size(output)
        return output

    def init_hidden(self, batch_size, device):
        """ 
        At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The shapes are (num_layers, batch_size, hidden_size)
        """
        return (torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device),
                torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device))