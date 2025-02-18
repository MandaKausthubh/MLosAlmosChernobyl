import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================
class AttentionBlock(nn.Module):

    def __init__(self, D, H = 1) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.V = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.K = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.Q = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.SM = nn.Softmax(-1)

    def forward(self, x):
        answer = []
        batchSize, SeqLen, Dim = x.shape
        head_size = Dim // self.H
        x = x.view(batchSize, self.H, SeqLen, head_size)
        for i in range(self.H):
            y = x[:, i, :, :]
            val = self.V[i](y)
            key = self.K[i](y)
            query = self.Q[i](y)

            # calculating the Attention values
            attention = self.SM(
                (torch.matmul(query, key.transpose(-2, -1)))/
                    torch.sqrt(torch.tensor(self.D/self.H, dtype=torch.float32))
            )
            answer.append(attention @ val)
        return torch.cat(answer, dim=-1)


# ========================================================
class MaskedAttentionBlock(nn.Module):

    def __init__(self, D, H = 1) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.V = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.K = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.Q = nn.ModuleList([nn.Linear(D//H, D//H) for _ in range(self.H)])
        self.SM = nn.Softmax(-1)

    def forward(self, x):
        answer = []
        batchSize, SeqLen, Dim = x.shape
        head_size = Dim // self.H
        x = x.view(batchSize, self.H, SeqLen, head_size)
        for i in range(self.H):
            y = x[:, i, :, :]
            val = self.V[i](y)
            key = self.K[i](y)
            query = self.Q[i](y)

            # calculating the Attention values
            attention = self.SM(
                torch.triu(torch.ones(SeqLen, SeqLen), diagonal=1).to(x.device) * float('-inf') + 
                (torch.matmul(query, key.transpose(-2, -1)))/
                    torch.sqrt(torch.tensor(self.D/self.H, dtype=torch.float32)),
            )
            answer.append(attention @ val)
        return torch.cat(answer, dim=-1)



# ========================================================
# Defining an Encoder Block
class Encoder(nn.Module):

    def __init__(self, N, M) -> None:
        super().__init__()
        self.N = N
        self.M = M
        self.attention = AttentionBlock(N)
        self.linear = nn.Linear(N, M)
        self.norm1 = nn.LayerNorm(N)
        self.norm2 = nn.LayerNorm(N)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = (self.ReLU(self.attention(self.norm1(x)) + x))
        return (self.linear(self.norm2(x)) + x)


# ========================================================
# Defining a Decoder

class Decoder(nn.Module):

    def __init__(self, N, M) -> None:
        super().__init__()
        self.N, self.M = N, M
