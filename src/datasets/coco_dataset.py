import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import json
import numpy as np
from PIL import Image
import nltk
import tqdm

from vocab import Vocab

class CoCoDataset(Dataset):
    '''
        A Pytorch dataset wrapper around the COCO Dataset
        Args: root:Path-like, the directory where you cloned the COCO api repo
    '''
    def __init__(self, coco_root, train_or_test, img_transform, batch_size, vocab_from_file=True, vocab_file='./vocab.pkl', year='2014') -> None:
        assert train_or_test in ['train', 'test'], "Mode must be either 'train' or 'test'"
        self.mode = train_or_test
        self.batch_size = batch_size
        self.transform = img_transform
        self.root = coco_root
        self.year = year
        if self.mode == 'train':
            if vocab_from_file == True: assert os.path.exists(vocab_file), "Vocab file does not exist"
            self.img_folder = os.path.join(self.root, 'images', f'train{self.year}')
            self.captions_file = os.path.join(self.root, 'annotations', f'captions_train{self.year}.json')
            self.coco = COCO(self.captions_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_toks = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[idx]]['caption'])) 
                    for idx in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_toks]
        else:
            assert os.path.exists(vocab_file), "Vocab file does not exist, make sure to generate one from training"
            print("Overriding batch_size to '1' for test mode")
            self.batch_size = 1 # Override the batch size
            self.vocab_from_file = True # Vocab must be from a file for test mode
            self.img_folder = os.path.join(self.root, 'images', f'test{self.year}')
            self.captions_file = os.path.join(self.root, 'annotations', f'image_info_test{self.year}.json')
            test_info = json.loads(open(self.captions_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
        self.vocab = Vocab(vocab_file)


    def __getitem__(self, index):
        if self.mode == 'train':
            # Find correct image and captions
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            img_name = self.coco.loadImgs(img_id)[0]['file_name']
            img_path = os.path.join(self.img_folder, img_name)
            img = Image.open(img_path).convert('RGB')

            # Tranform image to the size and shape we want
            img = self.transform(img)

            # Convert caption to tensor according to our vocab
            tokens = nltk.tokenize_word(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))

            return img, torch.Tensor(caption).long()
        else: 
            # Find correct image
            img_path = os.path.join(self.img_folder, self.paths[index])

            img = Image.open(img_path).convert('RGB')
            orig_img = np.array(img)
            tensor_img = self.transform(img)

            return orig_img, tensor_img

    def __len__(self):
        return len(self.ids) if self.mode == 'train' else len(self.paths)

    def get_train_indices(self):
        '''
            Randomly samples a batch_size worth of captions all of length cap_length
        '''
        assert self.mode == True, "Must be in training mode to use this function"
        cap_length = np.random.choice(self.caption_lengths)
        all_idxs = np.where([self.caption_lengths[i] == cap_length 
                    for i in np.arange(len(self.caption_lengths))])[0]
        return list(np.random.choice(all_idxs, size=self.batch_size))