import math
import torch
import torch.nn as nn
import torchvision.transforms as T

class EmbeddedPatches(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.patch_size = 16
        self.linear_proj = nn.Linear(768, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1+196, d_model))

    def forward(self, images):
        batch_tensor = images
        patches = batch_tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(batch_tensor.shape[0], -1, 3 * self.patch_size * self.patch_size)

        patch_embedding = self.linear_proj(patches)
        batch_size = patch_embedding.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        add_cls = torch.cat((cls_tokens, patch_embedding), dim=1)
        input = add_cls + self.pos_embedding
        return input


class Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.norm(x)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(param)

    def forward(self, x):
        Q = torch.matmul(x, self.W_Q)
        K = torch.matmul(x, self.W_K)
        V = torch.matmul(x, self.W_V)

        b, s, _ = Q.shape
        Q = Q.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(b, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(b, -1, self.num_heads, self.d_head).transpose(1, 2)

        QK_matmul = torch.matmul(Q, K.transpose(-2, -1))
        scale = QK_matmul / math.sqrt(self.d_head)

        softmax = torch.softmax(scale, dim=-1)
        SV_matmul = torch.matmul(softmax, V)
        concat = SV_matmul.transpose(1, 2).contiguous().view(b, s, -1)
        linear = torch.matmul(concat, self.W_O)
        return linear


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = d_model * 4
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self._reset_parameters()
        self.gelu = nn.GELU()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x):
        linear1 = torch.matmul(x, self.W1) + self.b1
        gelu = self.gelu(linear1)
        linear2 = torch.matmul(gelu, self.W2) + self.b2
        return linear2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multiatt = MultiHeadAttention(d_model, num_heads)
        self.norm1 = Norm(d_model)
        self.mlp = MLP(d_model)
        self.norm2 = Norm(d_model)

    def forward(self, encoder_input):
        norm1_out = self.norm1(encoder_input)
        multiatt_out = self.multiatt(norm1_out)
        multiatt_out = multiatt_out + encoder_input
        norm2_out = self.norm2(multiatt_out)
        mlp_out = self.mlp(norm2_out)
        mlp_out = mlp_out + multiatt_out
        return mlp_out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embed = EmbeddedPatches(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.encoder:
            x = layer(x)
        cls_token = x[:, 0, :]
        cls_token = self.norm(cls_token)
        out = self.mlp_head(cls_token)
        return out
