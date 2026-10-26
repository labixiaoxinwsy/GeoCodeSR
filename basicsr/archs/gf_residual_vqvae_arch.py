import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x) * 0.2


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, embed_dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, embed_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(embed_dim),
            ResBlock(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.body(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=512, embed_dim=64, commitment_weight=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, self.embed_dim)
        codebook = self.embedding.weight

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(codebook ** 2, dim=1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.codebook_size).type_as(z_flat)
        z_q = torch.matmul(encodings, codebook).view_as(z_perm)

        commitment_loss = F.mse_loss(z_q.detach(), z_perm)
        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        codebook_usage = torch.mean((avg_probs > 0).float())
        dead_code_ratio = torch.mean((avg_probs == 0).float())
        indices = indices.view(z.shape[0], z.shape[2], z.shape[3])

        return z_q, vq_loss, commitment_loss, indices, perplexity, codebook_usage, dead_code_ratio, avg_probs


class ResidualDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, embed_dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class GFResidualVQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        embed_dim=64,
        codebook_size=512,
        commitment_weight=0.25,
    ):
        super().__init__()
        self.encoder = ResidualEncoder(in_channels, base_channels, embed_dim)
        self.quantizer = VectorQuantizer(codebook_size, embed_dim, commitment_weight)
        self.decoder = ResidualDecoder(out_channels, base_channels, embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        (quantized, vq_loss, commitment_loss, indices, perplexity, codebook_usage,
         dead_code_ratio, code_probabilities) = self.quantizer(z)
        recon = self.decoder(quantized)
        return {
            'recon': recon,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'indices': indices,
            'perplexity': perplexity,
            'codebook_usage': codebook_usage,
            'dead_code_ratio': dead_code_ratio,
            'code_probabilities': code_probabilities,
        }
