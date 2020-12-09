import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size=512, num_layers=1, device='cuda'):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.device = device

        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.n_layers)
        self.to_out_size = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions):
        '''
        Args:
        features, image features from CNNEmbedder: shape -> (batch_size, embed_size)
        captions, captions for their respective images: shape -> (batch_size, caption_length)
        output: shape -> (batch_size, caption_length, vocab_size)
        '''
        captions = captions[:, :-1] # Remove the <end> word from the captions
        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size)
        embeddings = self.word_embedding(captions).view(-1, batch_size, self.embed_size) # shape: (caption_len, batch_size, embed_size)
        output = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        output, self.hidden = self.gru(output, self.hidden)
        output = self.to_out_size(output)
        return output

    def init_hidden(self, batch_size):
        """ 
        At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        Returns: initialized hidden layer, shape -> (num_layers, batch_size, hidden_size)
        """
        return torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(self.device)