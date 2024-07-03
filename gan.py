import numpy as np
import torch
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)

class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        self.pipeline = nn.Sequential(
            # nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(ResidualBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        x = self.pipeline(x)
        x = nn.functional.pad(x, (1, 0, 1, 0, 1, 0))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class Attention_block(nn.Module):
    #Adapted from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, verbose=False):
        super(Generator, self).__init__()
        self.verbose = verbose

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        x = self.first_layer(x)
        skip_connections = []
        if self.verbose:
            print("after first layer", x.shape)
        for i, d in enumerate(self.downs):
            skip_connections.append(x)
            x = d(x)
            if self.verbose:
                print("down" + str(i), x.shape)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))
            if self.verbose:
                print("bottlneck", x.shape)

        for i in range(len(self.ups)):
            if i == 0:
                #   print(x.shape, x_prev.shape)
                u = self.ups[i]
                concat = torch.cat((x, x_prev), dim=1)
                x = u(concat)
                if self.verbose:
                    print("up" + str(i), x.shape)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)
                concat = torch.cat((x, skip_connections[i - 1]), dim=1)
                #      print("--", concat.shape)
                x = u(concat)
                if self.verbose:
                    print("up" + str(i), x.shape)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

class AttentionGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        #adapated from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
        super(AttentionGenerator, self).__init__()

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.attentions = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.attentions.append(Attention_block(F_g=num_features[-1],F_l=num_features[-1], F_int=num_features[-i - 2] * 2))
            else:
                self.attentions.append(Attention_block(F_g=num_features[-i - 2] * 2, F_l=num_features[-i - 2] * 2, F_int=num_features[-i - 2]))


        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.first_layer(x)
        skip_connections = []
        for d in self.downs:
            skip_connections.append(x)
            x = d(x)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))

        for i in range(len(self.ups)):
            if i == 0:
                #   print(x.shape, x_prev.shape)
                u = self.ups[i]
                attention_out = self.attentions[i](x, x_prev)
                concat = torch.cat((x, attention_out), dim=1)
                x = u(concat)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)

                attention_out = self.attentions[i](x, skip_connections[i - 1])
                concat = torch.cat((x, attention_out), dim=1)
                #      print("--", concat.shape)
                x = u(concat)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

class Discriminator(nn.Module):
    def __init__(self, in_features=3, last_conv_kernalsize=4, verbose=False):
        super(Discriminator, self).__init__()

        self.verbose = verbose

        num_features = [in_features, 16, 32, 64, 128]

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.last_layer = nn.Sequential(
            nn.Conv3d(num_features[-1], 1, last_conv_kernalsize, 1, 1),
            #nn.Sigmoid()
        )

    def forward(self, x, alt_cond, oars=None):
        x = torch.cat((x, alt_cond), dim=1)
        if oars is not None:
            x = torch.cat((x, oars), dim=1)

        for d in self.downs:
            x = d(x)

        orig_shape = x.shape
        if self.verbose:
            print("before last layer", x.shape)
        x = self.last_layer(x)
        if self.verbose:
            print("after last layer", x.shape)

        # pad to original shape
        # if x.shape != orig_shape:
        #     difference = np.array(orig_shape) - np.array(x.shape)
        #     #        print(difference)
        #     x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
        #     if self.verbose:
        #         print("x", x.shape)

        return x

if __name__ == '__main__':
    model = AttentionGenerator(3, 1)
    #print(model.first_layer)
    #model = Discriminator(7, True)
    x = torch.randn((1, 2, 92, 152, 152))
    y = torch.randn((1, 1, 92, 152, 152))


    #
    device = torch.device('cuda:0')
    x = x.to(device)
    y = y.to(device)
    model.to(device)

    out = model(x, y)

