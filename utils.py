import torch
import numpy as np
import torch.nn as nn
import math

class Patchify(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.patch_size = config['patch_size']
    self.num_channels = config['num_channels']
    self.hidden_size = config['hidden_size']

    # Each patch has dimension = C * P^2
    patch_dim = self.num_channels * (self.patch_size ** 2)

    self.projection = nn.Linear(patch_dim, self.hidden_size)


  def forward(self,x):
    b,c,h,w = x.shape
    if h%self.patch_size != 0 or w%self.patch_size != 0:
      raise ValueError("Wrong Patch size or image dim, fix that")
    p = self.patch_size
    x = x.view(b, c, h / p, p, w / p, p)
    x = x.reshape(b, -1, c * p * p)
    x = self.proj(x)                         # (b, num_patches, hidden_size)

    return x

class Patchify_Conv(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.patch_size = config['patch_size']
    self.num_channels = config['num_channels']
    self.hidden_size = config['hidden_size']
    self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x):
    b,c,h,w = x.shape
    if h%self.patch_size != 0 or w%self.patch_size != 0:
      raise ValueError("Wrong Patch size or image dim, fix that")
    x = self.projection(x)
    x = x.flatten(2).transpose(1, 2)

    return x



class ViTPatchEmbed(nn.Module):

  def __init__(self, config):
    super().__init__()
    if config['patchify_method'] == "conv":
      self.patchify = Patchify_Conv(config)
    else:
      self.patchify = Patchify(config)
    self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
    self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
    self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, config["hidden_size"]))
    self.dropout = nn.Dropout(config["hidden_dropout_prob"])

  def forward(self, x):
    x = self.patchify(x)
    b, n, _ = x.shape
    cls_tokens = self.cls_token.expand(b, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.position_embeddings
    x = self.dropout(x)

    return x
