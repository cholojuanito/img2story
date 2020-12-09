import nltk
import os
from numpy import argsort
import pickle as pkl
from pycocotools.coco import COCO
from collections import OrderedDict, Counter

class Vocab(object):
    '''
        A class that makes it easy to build a vocabulary to be used in the 
        RNNDecoder and other classes. It also allows for new words to be
        added if the caption they appear in meets a certain threshold.
    '''
    def __init__(self, file_path, vocab_from_file, vocab_threshold, annotations_file_path, start_word='<start>', end_word='<end>', unknown_word='<unk>') -> None:
        if vocab_from_file == True: assert os.path.exists(file_path), "Vocab file does not exist"
        self.vocab_file_path = file_path # Contains the word2idx and idx2word dictionaries if already saved
        self.vocab_from_file = vocab_from_file # Whether or not to use the vocab file given, if False vocab_file_path will just be written to not read
        self.vocab_threshold = vocab_threshold
        self.annotations_file_path = annotations_file_path # Contains the words that are used to build the vocab the first time
        self.start_word = start_word
        self.end_word = end_word
        self.unknown_word = unknown_word
        self._get_vocab()

    def __getitem__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def _init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def _get_vocab(self):
        '''
            Load the vocabulary from file or build the vocabulary and save
            it to vocab_file_path
        '''
        if os.path.exists(self.vocab_file_path) and self.vocab_from_file:
            self.load_vocab()
            print(f'Vocab successfully loaded from {self.vocab_file_path}')
        else:
            self._build_vocab()
            self.save_vocab()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary from scratch
        """
        self._init_vocab()
        self._add_word(self.start_word)
        self._add_word(self.end_word)
        self._add_word(self.unknown_word)
        self._add_captions()

    def _add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def _add_captions(self):
        coco = COCO(self.annotations_file_path)
        cntr = Counter()
        ids = coco.anns.keys()
        print('Tokenizing captions...')
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            cntr.update(tokens)

            if i % 50000 == 0:
                print(f'{i}/{len(ids)} tokens')

        words = [word for word, count in cntr.items() if count >= self.vocab_threshold]

        print('Adding words to vocabulary')
        for w in words:
            self._add_word(w)

    def save_vocab(self) -> None:
        """
        Save a vocabulary to a file
        """
        with open(self.vocab_file_path, 'wb') as f:
            pkl.dump(self, f)

    def load_vocab(self) -> None:
        '''
        Load a vocabulary from a file
        '''
        with open(self.vocab_file_path, 'rb') as f:
            vocab = pkl.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
        