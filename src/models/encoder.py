import torch
import torch.nn as nn
import torchvision

class CNNEncoder(nn.Module):
    def __init__(self, embedding_size) -> None:
        super(CNNEncoder, self).__init__()
        resnet = torchvision.models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        old_shape = self.net.fc.in_features
        layers = list(resnet.children())[:-1] # Remove the fully connected last layer
        self.net = nn.Sequential(*layers)
        # Use embedding and batch norm instead of fully connected layer
        self.embed = nn.Linear(old_shape, embedding_size)
        # Initialize the weights and biases
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, x):
        feats = self.net(x)
        feats = feats.view(feats.shape[0], -1)
        return self.batch_norm(self.embed(feats))