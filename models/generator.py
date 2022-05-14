import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
from .modules import Conv1d


class Generator(nn.Module):
    def __init__(self,
                 in_channels=768,
                 z_channels=128,
                 n_mlp=4):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels

        # Style mapping
        self.style = nn.Sequential()
        for i in range(n_mlp):
            self.style.add_module('mapping network liner: {}'.format(i), nn.Linear(z_channels, z_channels))
            self.style.add_module('mapping network relu: {}'.format(i), nn.LeakyReLU(0.2))

        # upsample factor == 13500
        self.preprocess = Conv1d(in_channels, 768, kernel_size=3)
        self.gblocks = nn.ModuleList ([
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 3),
            GBlock(384, 384, z_channels, 3),
            GBlock(384, 192, z_channels, 3),
            GBlock(192, 192, z_channels, 5),
            GBlock(192, 192, z_channels, 5),
            GBlock(192, 96, z_channels, 5)
        ])
        self.postprocess = nn.Sequential(
            Conv1d(96, 1, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, inputs, z):
        inputs = self.preprocess(inputs)
        z = self.style(z)
        outputs = inputs
        for (i, layer) in enumerate(self.gblocks):
            outputs = layer(outputs, z)
        outputs = self.postprocess(outputs)

        return outputs

class GBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 z_channels,
                 upsample_factor):
        super(GBlock, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.z_channels = z_channels
        self.upsample_factor = upsample_factor

        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels, z_channels)
        self.first_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            Conv1d(in_channels, hidden_channels, kernel_size=3)
        )

        self.condition_batchnorm2 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.second_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        )

        self.residual1 = nn.Sequential(
           UpsampleNet(in_channels, in_channels, upsample_factor),
           Conv1d(in_channels, hidden_channels, kernel_size=1)
        )

        self.condition_batchnorm3 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.third_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4)
        )

        self.condition_batchnorm4 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.fourth_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=8)
        )

    def forward(self, condition, z):
        inputs = condition

        outputs = self.condition_batchnorm1(inputs, z)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, z)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, z)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, z)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs

class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor):

        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(input_size, output_size, upsample_factor * 2,
                                   upsample_factor, padding=upsample_factor // 2)
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs

class ConditionalBatchNorm1d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, z_channels=128):
      super().__init__()

      self.num_features = num_features
      self.z_channels = z_channels
      self.batch_nrom = nn.BatchNorm1d(num_features, affine=False)

      self.layer = spectral_norm(nn.Linear(z_channels, num_features * 2))
      self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.layer.bias.data.zero_()             # Initialise bias at 0

      self.add_noise = NoiseInjection()

    def forward(self, inputs, style, noise=None):
      # inputs 进行 Batch Normalization
      # style 进行 Spectral Normalization
      #################
      #    添加噪声
      # 1. 是否采用广播
      # 2. 是直接相加还接接一个全连接层
      # 3. 添加噪声的时机

      outputs = self.add_noise(inputs, noise)
      outputs = self.batch_nrom(outputs)
      gamma, beta = self.layer(style).chunk(2, 1)
      gamma = gamma.view(-1, self.num_features, 1)
      beta = beta.view(-1, self.num_features, 1)

      outputs = gamma * outputs + beta

      return outputs

class AdaIN(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, z_channels=128):
      super().__init__()

      self.num_features = num_features
      self.z_channels = z_channels
      self.batch_nrom = nn.BatchNorm1d(num_features, affine=False)

      self.layer = spectral_norm(nn.Linear(z_channels, num_features * 2))
      self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.layer.bias.data.zero_()             # Initialise bias at 0

      self.add_noise = NoiseInjection()

    def forward(self, inputs, style, noise=None):
      # inputs 进行 Batch Normalization
      # style 进行 Spectral Normalization
      #################
      #    添加噪声
      # 1. 是否采用广播
      # 2. 是直接相加还接接一个全连接层
      # 3. 添加噪声的时机

      # outputs = self.add_noise(inputs, noise)
      outputs = self.batch_nrom(outputs)
      gamma, beta = self.layer(style).chunk(2, 1)
      gamma = gamma.view(-1, self.num_features, 1)
      beta = beta.view(-1, self.num_features, 1)

      outputs = gamma * outputs + beta

      return outputs

##############################################################
#                       version 1
#    https://github.com/manicman1999/StyleGAN-Tensorflow-2.0/blob/5993fb3e5414821337f458a6be7fc864831679ff/stylegan.py#L97
#
###############################################################

##############################################################
#                       version 2
#    https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L332
#
###############################################################

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, height, width = x.shape
            noise = x.new_empty(batch, height, width).normal_()

        return x + self.weight * noise


if __name__ == '__main__':
    generator = Generator(in_channels=768)
    vec = torch.randn(3, 768, 5)
    z = torch.normal(0, 1, size=(3, 128))

    output = generator(vec, z)
    print(output.shape)
