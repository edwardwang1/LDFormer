from __future__ import print_function


import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)

# adapted from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=-krCPxqhAKMc

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, input_shape=None):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost
        if len(input_shape) > 0:
            self._input_shape = torch.Size(input_shape)

        self.usage_threshold = 1e-9
        self.register_buffer('usage', torch.zeros(num_embeddings))

    def update_usage(self, encoding_indices):
        self.usage[encoding_indices] = self.usage[encoding_indices] + 1  # if code is used add 1 to usage
        self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        self.usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        rand_codes = torch.randperm(self._num_embeddings)[0:len(dead_codes)]
        with torch.no_grad():
            self._embedding.weight[dead_codes] = self._embedding.weight[rand_codes]

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        self._input_shape = input_shape
        #print("forward through VQ, input shape", input_shape)


        # Flatten input 
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        #Update usage
        self.update_usage(encoding_indices)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, input_shape=None, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        if len(input_shape) > 0:
            self._input_shape = torch.Size(input_shape)

        self.usage_threshold = 1e-9
        self.register_buffer('usage', torch.zeros(num_embeddings))

    def update_usage(self, encoding_indices):
        self.usage[encoding_indices] = self.usage[encoding_indices] + 1  # if code is used add 1 to usage
        self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        self.usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        rand_codes = torch.randperm(self._num_embeddings)[0:len(dead_codes)]
        with torch.no_grad():
            self._embedding.weight[dead_codes] = self._embedding.weight[rand_codes]

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        self._input_shape = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        self.update_usage(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWLC -> BCHWL
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])
        

        for m in self._layers:
            init_weights(m)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_downsample_layers, extraConv=False):
        super(Encoder, self).__init__()
        self.num_downsample_layers = num_downsample_layers
        self.doExtraConv= extraConv

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // (2**(self.num_downsample_layers - 1)),
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._bn_1 = nn.BatchNorm2d(num_hiddens // (2**(self.num_downsample_layers - 1)))

        self._downConvs = nn.ModuleList([nn.Conv2d(in_channels=num_hiddens // (2**(self.num_downsample_layers - 1 - m)),
                                                   out_channels=num_hiddens // (2**(self.num_downsample_layers - 2 -m)),
                                        kernel_size=4,stride=2, padding=1)
                                        for m in range(self.num_downsample_layers - 1)])

        self._bn_down = nn.ModuleList([nn.BatchNorm2d(num_hiddens // (2**(self.num_downsample_layers - 2 -m))) for m in range(self.num_downsample_layers - 1)])

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._bn_3 = nn.BatchNorm2d(num_hiddens)

        if self.doExtraConv:
            self._extraConv = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=[2, 2], stride=[2, 2], padding=[2, 2]) #This is to go from 16x16 to 10x10
            self._bn_extra = nn.BatchNorm2d(num_hiddens)
            # pass

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        init_weights(self._conv_1)
        for m in self._downConvs:
            init_weights(m)
        init_weights(self._conv_3)
        if self.doExtraConv:
            init_weights(self._extraConv)

    def forward(self, x_):
        x_ = self._conv_1(x_)
        x_ = self._bn_1(x_)
        for i in range(self.num_downsample_layers - 1):
            x_ = self._downConvs[i](x_)
            x_ = self._bn_down[i](x_)
            x_ = F.relu(x_)

        x_ = self._conv_3(x_)
        x_ = self._bn_3(x_)

        if self.doExtraConv:
            x_ = F.relu(x_)
            x_ = self._extraConv(x_)
            x_ = self._bn_extra(x_)

        x_ = self._residual_stack(x_)

        #print("encoder end shape", x_.shape)
        return x_


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_downsample_layers, extraConv):
        super(Decoder, self).__init__()
        self.num_downsample_layers = num_downsample_layers
        self.doExtraConv = extraConv

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._bn_1 = nn.BatchNorm2d(num_hiddens)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        if self.doExtraConv:
            self._extraUpConv = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=[2, 2], stride=[2, 2], padding=[2, 2]) #This is to go from 16x16 to 10x10
            self._bn_extra = nn.BatchNorm2d(num_hiddens)
            #pass
        self._upConvs = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_hiddens // 2 ** (m),
                                                out_channels=num_hiddens // 2 ** (m + 1),
                                                kernel_size=4,
                                                stride=2, padding=1)
                                                for m in range(self.num_downsample_layers - 1)])

        self._bn_up = nn.ModuleList([nn.BatchNorm2d(num_hiddens // 2 ** (m + 1)) for m in range(self.num_downsample_layers - 1)])

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens // 2 ** (self.num_downsample_layers - 1),
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

        init_weights(self._conv_1)
        for m in self._upConvs:
            init_weights(m)
        init_weights(self._conv_trans_1)
        if self.doExtraConv:
            init_weights(self._extraUpConv)


    def forward(self, x_):
        x_ = self._conv_1(x_)
        x_ = self._bn_1(x_)
        x_ = self._residual_stack(x_)
        if self.doExtraConv:
            x_ = self._extraUpConv(x_)
            x_ = self._bn_extra(x_)
            x_ = F.relu(x_)
        for i in range(self.num_downsample_layers - 1):
            x_ = self._upConvs[i](x_)
            x_ = self._bn_up[i](x_)
            x_ = F.relu(x_)
        x_ = self._conv_trans_1(x_)
        return x_

@dataclass
class VQVAEConfig:
    in_channels: int = 1
    num_hiddens: int = 4 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_residual_layers: int = 4
    num_residual_hiddens: int = 4
    num_embeddings: int = 64
    embedding_dim: int = 4
    num_downsample_layers: int = 5
    commitment_cost: float = 0.25
    decay: float = 0.
    extra_conv: bool = False
    input_shape: list[int] = field(default_factory=list)

# def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
#                  num_embeddings, embedding_dim, num_downsample_layers, commitment_cost, decay=0., extra_conv=False):


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(config.in_channels, config.num_hiddens,
                                config.num_residual_layers,
                                config.num_residual_hiddens, config.num_downsample_layers, config.extra_conv)
        self._pre_vq_conv = nn.Conv2d(in_channels=config.num_hiddens,
                                      out_channels=config.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if config.decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(config.num_embeddings, config.embedding_dim,
                                              config.commitment_cost, config.decay, config.input_shape)
        else:
            self._vq_vae = VectorQuantizer(config.num_embeddings, config.embedding_dim,
                                           config.commitment_cost, config.input_shape)
        self._decoder = Decoder(config.embedding_dim, config.in_channels,
                                config.num_hiddens,
                                config.num_residual_layers,
                                config.num_residual_hiddens, config.num_downsample_layers, config.extra_conv)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        #print("vq_input_shape", z.shape)
        loss, quantized, perplexity, _, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def encode_to_c(self, x):
        x = self._encoder(x)
        x = self._pre_vq_conv(x)
        _, quantized, _, _, indices = self._vq_vae(x)
        if len(indices.shape) > 2:
            indices = indices.view(x.shape[0], -1)
        return quantized, indices
    
    def encode_to_c_batch(self, x):
        x = self._encoder(x)
        x = self._pre_vq_conv(x)
        _, quantized, _, _, indices = self._vq_vae(x)
        return quantized, indices.view(x.shape[0], -1)

    
    def decode_from_c(self, encoding_indices):
        encodings = torch.zeros(encoding_indices.shape[0], self._vq_vae._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._vq_vae._embedding.weight).view(self._vq_vae._input_shape).permute(0, 3, 1, 2).contiguous()
        x_recon = self._decoder(quantized)
        return x_recon

class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)


class Discriminator(nn.Module):
    def __init__(self, in_features=1, verbose=False):
        super(Discriminator, self).__init__()

        self.verbose = verbose

        num_features = [in_features, 2, 4]

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.last_layer = nn.Sequential(
            nn.Conv2d(num_features[-1], 1, 4, 2, 1),
        )

    def forward(self, x):
        for d in self.downs:
            x = d(x)

        # orig_shape = x.shape
        if self.verbose:
            print("before last layer", x.shape)
        x = self.last_layer(x)
        if self.verbose:
            print("after last layer", x.shape)

        return x


if __name__ == '__main__':

    in_channels = 1

    num_hiddens = 64
    num_residual_hiddens = 4
    num_residual_layers = 4
    num_downsample_layers = 3

    embedding_dim = 16
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99
    extra_conv = True

    config = VQVAEConfig(in_channels=in_channels, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                         num_residual_hiddens=num_residual_hiddens, num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim, num_downsample_layers=num_downsample_layers,
                            commitment_cost=commitment_cost, decay=decay, extra_conv=extra_conv)

    model = VQVAE(config)
    #model = Discriminator(verbose=True)

    x = torch.randn((1, 1, 128, 128))

    device = torch.device('cuda:0')
    #device = torch.device('cpu')

    x = x.to(device)
    model = model.to(device)

    y = model.encode_to_c(x)
    print(y[1].shape)

    out = model(x)
    print(out[1].shape)

    model._vq_vae.random_restart()
    model._vq_vae.reset_usage()

    d = Discriminator()
    d = d.to(device)
    disc_out = d(x)
    print(disc_out.shape)

