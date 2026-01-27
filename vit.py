import torch
import numpy as np
import torch.nn as nn
import math
from utils import Patchify, Patchify_Conv, ViTPatchEmbed

class ViTAttentionHead(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.hidden_size = config['hidden_size']
    self.attn_head_size = config['hidden_size'] // config['num_attention_heads']
    self.bias = config['bias'] # added just for the sake of experimentation
    self.w_query = nn.Linear(self.hidden_size, self.attn_head_size, bias=self.bias)
    self.w_key = nn.Linear(self.hidden_size, self.attn_head_size, bias=self.bias)
    self.w_value = nn.Linear(self.hidden_size, self.attn_head_size, bias=self.bias)
    self.dropout = nn.Dropout(config['hidden_dropout_prob'])


  def forward(self, x):
    q = self.w_query(x)
    k = self.w_key(x)
    v = self.w_value(x)

    attn_matrix = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
    attn_matrix = torch.softmax(attn_matrix, dim=-1)
    attn_matrix = self.dropout(attn_matrix)
    attn_output = torch.matmul(attn_matrix, v)

    return attn_matrix, attn_output


class ViTMultiHeadAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.hidden_size = config["hidden_size"]
    self.num_attention_heads = config["num_attention_heads"]
    self.attention_head_size = self.hidden_size // self.num_attention_heads
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.heads = nn.ModuleList([])
    for _ in range(self.num_attention_heads):
        head = ViTAttentionHead(config)
        self.heads.append(head)
    self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
    self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

  def forward(self, x, output_attentions=False):
      attention_outputs = [head(x) for head in self.heads]  # list of tuples (attn_matrix, attn_output)
      attention_output = torch.cat([attn_output for _, attn_output in attention_outputs], dim=-1)
      attention_output = self.output_projection(attention_output)
      attention_output = self.output_dropout(attention_output)
      attention_probs = torch.stack([attn_matrix for attn_matrix, _ in attention_outputs], dim=1)

      return attention_output, attention_probs

class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.hidden_size = config['hidden_size']
    self.intermediate_size = config['intermediate_size']
    self.dropout_prob = config['hidden_dropout_prob']
    self.layer1 = nn.Linear(self.hidden_size, self.intermediate_size)
    self.activation = nn.GELU()
    self.layer2 = nn.Linear(self.intermediate_size, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_prob)

  def forward(self, x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    x = self.dropout(x)

    return x

class ViTTransformerBlock(nn.Module):

  def __init__(self,config):
    super().__init__()
    self.attention = ViTMultiHeadAttention(config)
    self.ln_1 = nn.LayerNorm(config["hidden_size"])
    self.mlp = MLP(config)
    self.ln_2 = nn.LayerNorm(config["hidden_size"])

  def forward(self, x):
    attn_output, attn_probs = self.attention(self.ln_1(x))
    x = x + attn_output
    mlp_output = self.mlp(self.ln_2(x))
    x = x + mlp_output
    return x, attn_probs

class ViTEncoder(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.blocks = nn.ModuleList([])
    for _ in range(config["num_hidden_layers"]):
      block = ViTTransformerBlock(config)
      self.blocks.append(block)

  def forward(self, x):
    all_attentions = []
    for block in self.blocks:
      x, attn_probs = block(x)
      all_attentions.append(attn_probs)
    return x, all_attentions


class ViT_cifar10(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.image_size = config["image_size"]
    self.hidden_size = config["hidden_size"]
    self.num_classes = config["num_classes"]
    self.patch_embed = ViTPatchEmbed(config)
    self.encoder = ViTEncoder(config)
    self.head = nn.Linear(self.hidden_size, self.num_classes)
  def forward(self, x):
    embedding_out = self.patch_embed(x)
    encoder_out, attn_weights = self.encoder(embedding_out)
    cls_token = encoder_out[:, 0]
    logits = self.head(cls_token)
    return logits, attn_weights


