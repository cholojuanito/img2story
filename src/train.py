import os
import nltk
# nltk.download('punkt') # Uncomment if you need to download this
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from datasets.coco_dataset import CoCoDataset
from models.decoder import RNNDecoder
from models.encoder import CNNEncoder
from vocab import Vocab
from training.coco_trainer import COCOCaptionTrainer


def train(n_epochs, lr, vocab, dataloader, embed_size=512, hidden_size=512, n_layers=1, device='cuda'):
    vocab_size = len(vocab)
    # Initialize Encoder and Decoder
    encoder = CNNEncoder(embed_size).to(device)
    decoder = RNNDecoder(embed_size, vocab_size, hidden_size=hidden_size, num_layers=n_layers).to(device)
    # Initialize optimizer
    all_params = list(decoder.parameters())  + list(encoder.embed.parameters()) + list(encoder.batch_norm.parameters())
    optimizer = torch.optim.Adam(params = all_params, lr = lr)
    loss_func = nn.CrossEntropyLoss()

    decoder_input_params = {
        'embed_size' : embed_size, 
        'hidden_size' : hidden_size, 
        'num_layers' : n_layers,
        'lr' : lr,
        'vocab_size' : vocab_size
    }

    trainer = COCOCaptionTrainer()
    trainer.encoder = encoder
    trainer.decoder = decoder
    trainer.optimizer = optimizer
    trainer.loss_func = loss_func
    trainer.dataloader = dataloader
    trainer.vocab_size = vocab_size

    losses = trainer.train(n_epochs)

def predict():
    pass

def get_vocab():
    pass

def main():
    # Define local paths
    coco_api_root = 'D:/dev/data/cocoapi'
    year = '2014' # What version/year of the COCO dataset to use
    train_vocab_file_path = f'./vocab{year}_train.pkl'
    test_vocab_file_path = f'./vocab{year}_test.pkl'
    annotations_folder = os.path.join(coco_api_root, 'annotations')
    # imgs_folder = os.path.join(coco_api_root, 'images')
    train_ann_file_path = os.path.join(annotations_folder, f'captions_train{year}.json')
    test_ann_file_path = os.path.join(annotations_folder, f'image_info_test{year}.json')

    # Setup other needed params
    train_str = 'train'
    test_str = 'test'
    num_epochs = 1
    lr = 1e-3
    num_workers = 5
    embeddding_size = 512 # Length of the word embedding
    hidden_size = 512 # Size of hidden layer in the decoder
    num_layers = 3 # Number of layers to stack the GRU in the decoder
    batch_size = 256
    img_size = 256
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), # normalize image for pre-trained resnet that is used
                            (0.229, 0.224, 0.225))
    ])
    vocab_threshold = 6
    vocab_from_file = True # Set this to False if you don't already have a vocab file made


    # Initialize vocabs and datasets
    train_vocab = Vocab(train_vocab_file_path, vocab_from_file, vocab_threshold, train_ann_file_path)
    # test_vocab = Vocab(test_vocab_file_path, vocab_from_file, vocab_threshold, test_ann_file_path)
    train_dataset = CoCoDataset(coco_api_root, train_str, img_transform, batch_size, year=year, vocab=train_vocab)
    # test_dataset = CoCoDataset(coco_api_root, test, img_transform, batch_size, year=year, vocab_from_file=vocab_from_file, vocab_file=vocab_file_path)
    
    
    # Initialize data loaders
    idxs = train_dataset.get_train_indices()
    initial_sampler = data.sampler.SubsetRandomSampler(indices=idxs)
    train_loader = data.DataLoader(train_dataset, 
                                    num_workers=num_workers, 
                                    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=train_dataset.batch_size,
                                                                              drop_last=False))
    # test_loader = data.DataLoader(test_dataset, 
    #                                 batch_size=test_dataset.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=num_workers)

    train(num_epochs, lr, train_vocab, train_loader, embed_size=embeddding_size, hidden_size=hidden_size, n_layers=num_layers)

    # Explore vocab
    # sample_caption = 'A person doing a trick xxxx on a rail while riding a skateboard.'
    # sample_tokens = nltk.tokenize.word_tokenize(sample_caption.lower()) 
    # sample_captions = []
    # start_word  = train_vocab.start_word
    # end_word = train_vocab.end_word
    # sample_tokens.insert(0 , start_word)
    # sample_tokens.append(end_word)
    # sample_captions.extend([train_vocab(token) for token in sample_tokens])

    # sample_captions = torch.Tensor(sample_captions).long()
    # print('Sample tokens and the idx values of those tokens in word2idx' , '\n')
    # print(sample_tokens) 
    # print(sample_captions)

    # # Check out first couple elements in word2idx in vocab 
    # print('First few vocab' , dict(list(train_vocab.word2idx.items())[:10]))
    # # Print the total number of keys in the word2idx dictionary
    # print('Total number of tokens in vocabulary:', len(train_vocab))


if __name__ == "__main__":
    main()