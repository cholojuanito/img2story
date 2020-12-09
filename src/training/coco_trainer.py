import math
import os
import sys
import torch
import torch.utils.data as data
from tqdm import tqdm

class COCOCaptionTrainer(object):
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.loss_func = None
        self.dataloader = None
        self.vocab_size = None
        self.device = 'cuda'
        self.epoch_count = 0

        self.loop = None

    def increment_epoch_count(self) -> None:
        self.epoch_count += 1
        
    def train(self, num_epochs):
        losses = []
        
        for e in range(self.epoch_count, num_epochs):
            total_steps = math.ceil(len(self.dataloader.dataset) / self.dataloader.batch_sampler.batch_size)
            self.loop = tqdm(total=total_steps, position=0, leave=False)
            print(f"\nTraining epoch {e} of {num_epochs}")
            loss = self._train_epoch(self.dataloader)
            self.loop.close()
            self.increment_epoch_count()
            losses.append(loss)
            self.save_model_state()
        return losses

        
    def _train_epoch(self, dataloader):
        steps_per_epoch = math.ceil(len(dataloader.dataset) / dataloader.batch_sampler.batch_size)

        loss_sum = 0
        for step in range(steps_per_epoch):
            # Get new sample of indices for captions of a random length
            indices = dataloader.dataset.get_train_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices)
            dataloader.batch_sampler.sampler = new_sampler 
            # Get images and respective captions   
            images, captions = next(iter(dataloader))
            images, captions = images.to(self.device, non_blocking=True), captions.to(self.device, non_blocking=True)
            self.encoder, self.decoder = self.encoder.to(self.device, non_blocking=True) , self.decoder.to(self.device, non_blocking=True)

            # Train
            # Reset gradient
            self.encoder.zero_grad()    
            self.decoder.zero_grad()
            # Pass inputs through models
            features = self.encoder(images)
            out = self.decoder(features, captions) 
            # Calculate loss   
            loss = self.loss_func(out.view(-1, self.vocab_size), captions.view(-1))
            loss_sum += loss.detach().item()

            mem_allocated = torch.cuda.memory_allocated(0) / 1e9

            self.loop.set_description(f'Loss: {loss.detach().item()}, memory: {mem_allocated}')
            self.loop.update(1)

            # Compute gradient
            loss.backward()
            self.optimizer.step()

        return loss_sum / steps_per_epoch

    def save_model_state(self):
        model_state = {
            'epoch': self.epoch_count,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(model_state, f'coco-caption-{self.epoch_count}.state')
        # torch.save(self.encoder, os.path.join(save_dir, f'encoder_{self.epoch_count+1}.pkl'))
        # torch.save(self.decoder, os.path.join(save_dir, f'decoder_{self.epoch_count+1}.pkl'))


    def load_model_state(self, load_path):
        pass