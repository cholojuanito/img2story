from numpy import argsort
import pickle as pkl
from collections import OrderedDict

class Vocab:
    def __init__(self, file_path, start_word='<start>', end_word='<end>', unknown_word='<unk>') -> None:
        self.__dict__.update(locals())

    def _init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def build_vocab(self, tokens:list[str]) -> None:
        """
        Build a vocabulary
        tokens: list of sentences (pre-tokenized)
        """
        wordcount = OrderedDict()
        for cc in tokens:
            words = cc.split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1
        words = wordcount.keys()
        freqs = wordcount.values()
        sorted_idx = argsort(freqs)[::-1]

        worddict = OrderedDict()
        for idx, sidx in enumerate(sorted_idx):
            # Start at idx 2 so there is room for the end of sentence token "<eos>"
            # and the unknown token "<unk>"
            worddict[words[sidx]] = idx + 2

        self.dictionary = worddict
        self.word_count = wordcount

    def save_vocab(self, save_path) -> None:
        """
        Save a vocabulary
        """
        with open(save_path, 'wb') as f:
            pkl.dump(self.dictionary, f)
            pkl.dump(self.word_count, f)

    def load_vocab(self, file_path) -> None:
        with open(file_path, 'rb') as f:
            contents = pkl.load(f)
        
        self.dictionary = contents['dictionary']
        self.word_count = contents['word_count']