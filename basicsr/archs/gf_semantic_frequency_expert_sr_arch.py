import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.gf_residual_vqvae_arch import (
    ResidualEncoder, ResBlock as ResidualResBlock, VectorQuantizer)
from basicsr.archs.quantsr_arch import MultiScaleEncoder, ResBlock as QuanTexResBlock
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ['params_ema', 'params', 'model', 'state_dict']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], key
    return checkpoint, 'direct'


def strip_prefix(state_dict, prefixes):
    cleaned = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned[name] = value
    return cleaned


def pool_semantic(semantic_token):
    if semantic_token.ndim == 2:
        return semantic_token.float()
    if semantic_token.ndim == 3:
        return semantic_token.mean(dim=1).float()
    if semantic_token.ndim == 4:
        return semantic_token.flatten(2).mean(dim=2).float()
    raise ValueError(f'Unsupported semantic token shape: {tuple(semantic_token.shape)}')


def indices_stats(indices, codebook_size):
    encodings = F.one_hot(indices.reshape(-1), codebook_size).float()
    probabilities = encodings.mean(dim=0)
    perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + 1e-10)))
    usage = torch.mean((probabilities > 0).float())
    return perplexity, usage


def gaussian_kernel1d(sigma, device, dtype):
    radius = max(int(3.0 * float(sigma) + 0.5), 1)
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinate ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def gaussian_blur(feature, sigma):
    channels = feature.shape[1]
    kernel = gaussian_kernel1d(sigma, feature.device, feature.dtype)
    radius = kernel.numel() // 2
    pad_mode = 'reflect' if feature.shape[-2] > radius and feature.shape[-1] > radius else 'replicate'
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    feature = F.conv2d(F.pad(feature, (radius, radius, 0, 0), mode=pad_mode), kernel_x, groups=channels)
    return F.conv2d(F.pad(feature, (0, 0, radius, radius), mode=pad_mode), kernel_y, groups=channels)


def group_norm(channels):
    groups = min(32, int(channels))
    while int(channels) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-6, affine=True)


class SemanticResidualCodebook(nn.Module):
    """Retrieve Stage 1 residual codes from LR content and a frozen SatMAE token."""

    def __init__(self, in_channels=3, base_channels=64, embed_dim=64, codebook_size=512,
                 token_dim=1024, residual_scale=20.0, temperature=1.0, hard_in_eval=True):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.residual_scale = float(residual_scale)
        self.temperature = float(temperature)
        self.hard_in_eval = bool(hard_in_eval)

        self.target_encoder = ResidualEncoder(in_channels, base_channels, embed_dim)
        self.quantizer = VectorQuantizer(codebook_size, embed_dim)
        self.content_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualResBlock(base_channels),
            ResidualResBlock(base_channels),
        )
        self.semantic_norm = nn.LayerNorm(token_dim)
        self.semantic_proj = nn.Linear(token_dim, base_channels * 2)
        self.predictor = nn.Sequential(
            ResidualResBlock(base_channels),
            ResidualResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualResBlock(base_channels * 2),
            ResidualResBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, codebook_size, 1, 1, 0),
        )
        nn.init.zeros_(self.semantic_proj.weight)
        nn.init.zeros_(self.semantic_proj.bias)

    def load_residual_vq(self, path, freeze_encoder=True, freeze_codebook=True):
        checkpoint = torch.load(str(Path(path)), map_location='cpu')
        state_dict, key = extract_state_dict(checkpoint)
        encoder_state = {name[8:]: value for name, value in state_dict.items() if name.startswith('encoder.')}
        quantizer_state = {name[10:]: value for name, value in state_dict.items() if name.startswith('quantizer.')}
        if not encoder_state or not quantizer_state:
            raise RuntimeError(f'Residual VQ checkpoint is incomplete: {path}')
        self.target_encoder.load_state_dict(encoder_state, strict=True)
        self.quantizer.load_state_dict(quantizer_state, strict=True)
        if freeze_encoder:
            self.target_encoder.requires_grad_(False)
            self.target_encoder.eval()
        if freeze_codebook:
            self.quantizer.requires_grad_(False)
            self.quantizer.eval()
        print(f'[GeoCodeSR] residual VQ loaded from key={key}: {path}')

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.target_encoder.parameters()):
            self.target_encoder.eval()
        if not any(parameter.requires_grad for parameter in self.quantizer.parameters()):
            self.quantizer.eval()
        return self

    def forward(self, content_lr, semantic_token, target_residual=None,
                semantic_guidance=True, retrieval_mode='soft'):
        if retrieval_mode not in {'soft', 'hard'}:
            raise ValueError(f'Unsupported SRCR retrieval mode: {retrieval_mode}.')
        semantic = self.semantic_norm(pool_semantic(semantic_token))
        feature = self.content_stem(content_lr)
        modulation = self.semantic_proj(semantic) if semantic_guidance else torch.zeros(
            feature.shape[0], feature.shape[1] * 2, device=feature.device, dtype=feature.dtype)
        gamma, beta = modulation.chunk(2, dim=1)
        feature = feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        logits = self.predictor(feature)
        probability = torch.softmax(logits / max(self.temperature, 1e-6), dim=1)
        indices = torch.argmax(logits, dim=1)

        codebook = self.quantizer.embedding.weight
        soft_code = torch.einsum('bkhw,kc->bchw', probability, codebook)
        hard_code = F.embedding(indices, codebook).permute(0, 3, 1, 2).contiguous()
        code_feat = hard_code if retrieval_mode == 'hard' else soft_code

        confidence = probability.amax(dim=1, keepdim=True)
        entropy_raw = -torch.sum(probability * torch.log(probability + 1e-8), dim=1, keepdim=True)
        normalized_entropy = entropy_raw / math.log(float(self.codebook_size))
        perplexity, usage = indices_stats(indices, self.codebook_size)
        output = {
            'code_feat': code_feat,
            'soft_code_feat': soft_code,
            'code_indices': indices,
            'pred_indices': indices,
            'code_logits': logits,
            'code_prob': probability,
            'code_confidence': confidence,
            'code_entropy': normalized_entropy,
            'code_entropy_raw': entropy_raw,
            'normalized_entropy': normalized_entropy,
            'code_uncertainty': normalized_entropy,
            'pred_perplexity': perplexity,
            'pred_codebook_usage': usage,
        }

        # ResidualEncoder is a target-label path and is never used in evaluation/inference.
        if self.training and target_residual is not None:
            with torch.no_grad():
                target_z = self.target_encoder(target_residual * self.residual_scale)
                (_, _, _, target_indices, target_perplexity, target_usage,
                 _, _) = self.quantizer(target_z)
            output.update({
                'target_indices': target_indices.long(),
                'target_perplexity': target_perplexity,
                'target_codebook_usage': target_usage,
                'code_acc': (indices == target_indices).float().mean(),
            })
        return output


class ScaleConditionAligner(nn.Module):
    """Explicitly align code, semantic, confidence, and entropy to one decoder scale."""

    def __init__(self, embed_dim, token_dim, out_channels):
        super().__init__()
        self.out_channels = int(out_channels)
        self.code_proj = nn.Conv2d(embed_dim, out_channels, 1)
        self.semantic_norm = nn.LayerNorm(token_dim)
        self.semantic_proj = nn.Linear(token_dim, out_channels)

    def forward(self, code_feat, semantic_token, confidence, normalized_entropy, spatial_size):
        code = self.code_proj(code_feat)
        if code.shape[-2:] != spatial_size:
            code = F.interpolate(code, size=spatial_size, mode='bilinear', align_corners=False)
        semantic = self.semantic_proj(
            self.semantic_norm(pool_semantic(semantic_token)))[:, :, None, None]
        semantic = semantic.expand(-1, -1, *spatial_size)
        confidence = F.interpolate(confidence, size=spatial_size, mode='bilinear', align_corners=False)
        normalized_entropy = F.interpolate(
            normalized_entropy, size=spatial_size, mode='bilinear', align_corners=False)
        return {
            'code': code,
            'semantic': semantic,
            'confidence': confidence,
            'entropy': normalized_entropy,
        }


class GatedCodeSkipFusion(nn.Module):
    """Replace SimSFT with explicit skip and residual-code gates."""

    def __init__(self, channels, skip_channels=None):
        super().__init__()
        self.channels = int(channels)
        self.skip_proj = nn.Conv2d(skip_channels, channels, 1) if skip_channels else None
        context_channels = channels * (4 if skip_channels else 3) + 2
        gate_channels = channels * (2 if skip_channels else 1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(context_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            QuanTexResBlock(channels, channels, norm_type='gn', act_type='silu'),
            nn.Conv2d(channels, gate_channels, 1),
        )
        self.refine = nn.Sequential(
            QuanTexResBlock(channels, channels, norm_type='gn', act_type='silu'),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, -2.0)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feature, skip, condition, lambda_code, enabled=True,
                use_confidence=True, use_entropy=True):
        code = condition['code']
        semantic = condition['semantic']
        confidence = condition['confidence']
        entropy = condition['entropy']
        spatial_size = feature.shape[-2:]
        for name, value in condition.items():
            if value.shape[-2:] != spatial_size:
                raise ValueError(f'{name} condition {value.shape[-2:]} does not match feature {spatial_size}.')

        contexts = [feature]
        projected_skip = None
        if self.skip_proj is not None:
            if skip is None:
                raise ValueError('A skip feature is required by this fusion block.')
            projected_skip = self.skip_proj(skip)
            if projected_skip.shape[-2:] != spatial_size:
                raise ValueError(
                    f'Skip size {projected_skip.shape[-2:]} does not match decoder size {spatial_size}.')
            contexts.append(projected_skip)
        confidence = confidence if use_confidence else torch.zeros_like(confidence)
        entropy = entropy if use_entropy else torch.zeros_like(entropy)
        if not enabled:
            fused = feature if projected_skip is None else feature + projected_skip
            zeros = torch.zeros_like(feature)
            stats = {
                'skip_gate': None,
                'code_gate': zeros,
                'skip_gate_mean': feature.new_zeros(()),
                'code_gate_mean': feature.new_zeros(()),
                'confidence_mean': confidence.mean(),
                'entropy_mean': entropy.mean(),
            }
            return fused, zeros, stats
        contexts.extend([code, semantic, confidence, entropy])
        gates = torch.sigmoid(self.gate_net(torch.cat(contexts, dim=1)))

        if projected_skip is None:
            skip_gate = None
            code_gate = gates
            fused = feature + lambda_code * code_gate * code
        else:
            skip_gate, code_gate = gates.chunk(2, dim=1)
            fused = feature + skip_gate * projected_skip + lambda_code * code_gate * code
        fused = fused + self.refine(fused)
        code_delta = code_gate * code
        stats = {
            'skip_gate': skip_gate,
            'code_gate': code_gate,
            'skip_gate_mean': (feature.new_zeros(()) if skip_gate is None else skip_gate.mean()),
            'code_gate_mean': code_gate.mean(),
            'confidence_mean': confidence.mean(),
            'entropy_mean': entropy.mean(),
        }
        return fused, code_delta, stats


class DynamicPixelReassembly(nn.Module):
    """CARAFE reassembly with a zero-initialized semantic-code modulation branch."""

    def __init__(self, in_channels, out_channels, condition_channels, kernel_size=3,
                 scale=2, hidden_channels=64, normalize_kernel=True):
        super().__init__()
        if not normalize_kernel:
            raise ValueError('DynamicPixelReassembly requires a normalized spatial kernel.')
        self.kernel_size = int(kernel_size)
        self.scale = int(scale)
        self.channel_compressor = nn.Conv2d(in_channels, hidden_channels, 1)
        self.code_norm = group_norm(condition_channels)
        self.semantic_norm = group_norm(condition_channels)
        self.code_condition = nn.Conv2d(condition_channels, hidden_channels, 1)
        self.semantic_condition = nn.Conv2d(condition_channels, hidden_channels, 1)
        self.confidence_condition = nn.Conv2d(1, hidden_channels, 1)
        self.entropy_condition = nn.Conv2d(1, hidden_channels, 1)
        self.condition_fusion = nn.Conv2d(hidden_channels * 4, hidden_channels, 1)
        self.kernel_head = nn.Conv2d(
            hidden_channels, self.kernel_size ** 2 * self.scale ** 2, 3, 1, 1)
        self.post_reassembly = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            QuanTexResBlock(out_channels, out_channels, norm_type='gn', act_type='silu'),
            QuanTexResBlock(out_channels, out_channels, norm_type='gn', act_type='silu'),
        )
        nn.init.zeros_(self.condition_fusion.weight)
        nn.init.zeros_(self.condition_fusion.bias)
        nn.init.normal_(self.kernel_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.kernel_head.bias)

    def _fixed_center_kernel(self, feature):
        batch, _, height, width = feature.shape
        kernel = feature.new_zeros(
            batch, self.kernel_size ** 2, height * self.scale, width * self.scale)
        kernel[:, self.kernel_size ** 2 // 2] = 1.0
        return kernel

    def forward(self, feature, condition, dynamic=True, use_confidence=True, use_entropy=True):
        batch, _, height, width = feature.shape
        for name, value in condition.items():
            if value.shape[-2:] != (height, width):
                raise ValueError(
                    f'{name} condition {value.shape[-2:]} does not match reassembly input {(height, width)}.')

        if dynamic:
            compressed = self.channel_compressor(feature)
            confidence = condition['confidence'] if use_confidence else torch.zeros_like(condition['confidence'])
            entropy = condition['entropy'] if use_entropy else torch.zeros_like(condition['entropy'])
            condition_delta = self.condition_fusion(torch.cat([
                self.code_condition(self.code_norm(condition['code'])),
                self.semantic_condition(self.semantic_norm(condition['semantic'])),
                self.confidence_condition(confidence),
                self.entropy_condition(entropy),
            ], dim=1))
            logits = self.kernel_head(compressed + condition_delta)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Dynamic reassembly kernel logits contain NaN or Inf.')
            kernel = torch.softmax(
                F.pixel_shuffle(logits, self.scale).float(), dim=1).to(feature.dtype)
        else:
            kernel = self._fixed_center_kernel(feature)

        patches = F.unfold(
            feature, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(
            batch, feature.shape[1] * self.kernel_size ** 2, height, width)
        # Nearest expansion only aligns source neighborhoods to the output grid.
        patches = F.interpolate(patches, scale_factor=self.scale, mode='nearest')
        patches = patches.view(
            batch, feature.shape[1], self.kernel_size ** 2,
            height * self.scale, width * self.scale)
        reassembled = torch.sum(patches * kernel.unsqueeze(1), dim=2)
        output = self.post_reassembly(reassembled)
        kernel_entropy = -torch.sum(kernel * torch.log(kernel + 1e-8), dim=1).mean()
        smoothness = (
            torch.mean(torch.abs(kernel[:, :, 1:, :] - kernel[:, :, :-1, :]))
            + torch.mean(torch.abs(kernel[:, :, :, 1:] - kernel[:, :, :, :-1])))
        return output, kernel, kernel_entropy, smoothness


class FrequencyExpert(nn.Module):
    def __init__(self, channels, hidden_channels=64):
        super().__init__()
        self.body = nn.Sequential(
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, 3, 1, 1),
            QuanTexResBlock(hidden_channels, hidden_channels, norm_type='gn', act_type='silu'),
            QuanTexResBlock(hidden_channels, hidden_channels, norm_type='gn', act_type='silu'),
            group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature):
        return self.body(feature)


class SpatialRoutedFrequencyExperts(nn.Module):
    """Decoder-embedded low/mid/high experts with per-pixel semantic-code routing."""

    def __init__(self, channels, hidden_channels=64, sigma_low=2.0, sigma_high=1.0):
        super().__init__()
        self.sigma_low = float(sigma_low)
        self.sigma_high = float(sigma_high)
        self.experts = nn.ModuleList([
            FrequencyExpert(channels, hidden_channels) for _ in range(3)])
        self.single_expert = FrequencyExpert(channels, hidden_channels)
        self.feature_norm = group_norm(channels)
        self.code_norm = group_norm(channels)
        self.semantic_norm = group_norm(channels)
        router_hidden = min(int(hidden_channels), int(channels))
        self.router = nn.Sequential(
            nn.Conv2d(channels * 3 + 2, router_hidden, 1),
            nn.SiLU(inplace=True),
            QuanTexResBlock(router_hidden, router_hidden, norm_type='gn', act_type='silu'),
            nn.Conv2d(router_hidden, 3, 3, 1, 1),
        )
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, feature, condition, lambda_freq, mode='prior'):
        batch, _, height, width = feature.shape
        if mode not in {'disabled', 'single', 'equal', 'content', 'prior'}:
            raise ValueError(f'Unsupported PMEF mode: {mode}.')
        if mode == 'disabled':
            weights = feature.new_full((batch, 3, height, width), 1.0 / 3.0)
            zeros = tuple(torch.zeros_like(feature) for _ in range(3))
            diagnostics = {
                'delta': torch.zeros_like(feature),
                'weights': weights,
                'entropy': feature.new_tensor(math.log(3.0)),
                'load': feature.new_full((3,), 1.0 / 3.0),
                'hard_load': feature.new_full((3,), 1.0 / 3.0),
                'balance': feature.new_zeros(()),
                'router_z_loss': feature.new_zeros(()),
                'expert_outputs': zeros,
                'expert_norms': feature.new_zeros(3),
                'collapse_score': feature.new_tensor(1.0 / 3.0),
            }
            return feature, diagnostics

        if mode == 'single':
            delta = self.single_expert(feature)
            weights = feature.new_zeros((batch, 3, height, width))
            weights[:, 0] = 1.0
            norm = torch.sqrt(torch.mean(delta.float() ** 2) + 1e-12)
            diagnostics = {
                'delta': delta,
                'weights': weights,
                'entropy': feature.new_zeros(()),
                'load': feature.new_tensor([1.0, 0.0, 0.0]),
                'hard_load': feature.new_tensor([1.0, 0.0, 0.0]),
                'balance': feature.new_zeros(()),
                'router_z_loss': feature.new_zeros(()),
                'expert_outputs': (delta, torch.zeros_like(delta), torch.zeros_like(delta)),
                'expert_norms': torch.stack([norm, feature.new_zeros(()), feature.new_zeros(())]),
                'collapse_score': feature.new_ones(()),
            }
            return feature + lambda_freq * delta, diagnostics

        low = gaussian_blur(feature, self.sigma_low)
        high_blur = gaussian_blur(feature, self.sigma_high)
        bands = (low, high_blur - low, feature - high_blur)
        expert_outputs = tuple(expert(band) for expert, band in zip(self.experts, bands))
        if mode == 'equal':
            router_logits = None
            weights = feature.new_full((batch, 3, height, width), 1.0 / 3.0)
        else:
            if mode == 'content':
                code = torch.zeros_like(condition['code'])
                semantic = torch.zeros_like(condition['semantic'])
                confidence = torch.zeros_like(condition['confidence'])
                entropy = torch.zeros_like(condition['entropy'])
            else:
                code, semantic = condition['code'], condition['semantic']
                confidence, entropy = condition['confidence'], condition['entropy']
            router_input = torch.cat([
                self.feature_norm(feature), self.code_norm(code), self.semantic_norm(semantic),
                confidence, entropy], dim=1)
            router_logits = self.router(router_input)
            if not torch.isfinite(router_logits).all():
                raise FloatingPointError('Frequency router logits contain NaN or Inf.')
            weights = torch.softmax(router_logits.float(), dim=1).to(feature.dtype)
        stacked = torch.stack(expert_outputs, dim=1)
        delta = torch.sum(weights.unsqueeze(2) * stacked, dim=1)
        output = feature + lambda_freq * delta
        load = weights.mean(dim=(0, 2, 3))
        hard_indices = torch.argmax(weights, dim=1)
        hard_load = F.one_hot(hard_indices, num_classes=3).float().mean(dim=(0, 1, 2))
        balance = 3.0 * torch.sum(hard_load.detach() * load.float())
        router_z_loss = (feature.new_zeros(()) if router_logits is None else
                 torch.mean(torch.logsumexp(router_logits.float(), dim=1) ** 2))
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        expert_norms = torch.stack([
            torch.sqrt(torch.mean(value.float() ** 2) + 1e-12) for value in expert_outputs])
        diagnostics = {
            'delta': delta,
            'weights': weights,
            'entropy': entropy,
            'load': load,
            'hard_load': hard_load,
            'balance': balance,
            'router_z_loss': router_z_loss,
            'expert_outputs': expert_outputs,
            'expert_norms': expert_norms,
            'collapse_score': hard_load.max(),
        }
        return output, diagnostics


@ARCH_REGISTRY.register()
class GFSemanticFrequencyExpertSRNet(nn.Module):
    """Single-path GeoCodeSR with a rebuilt code-guided dynamic decoder."""

    def __init__(self, backbone_ckpt='experiments/pretrained_models/quantexsr_net_g_best.pth',
                 residual_vq_ckpt=None, in_channels=3, out_channels=3, scale=4,
                 gt_resolution=256, base_channels=64, embed_dim=64, codebook_size=512,
                 token_dim=1024, residual_scale=20.0, temperature=1.0, hard_in_eval=True,
                 use_original_backbone=True, use_semantic_codebook=True, use_dynamic_kernel=True,
                 use_frequency_experts=True, kernel_size=3, normalize_kernel=True,
                 srcr_semantic_guidance=True, srcr_retrieval='soft', use_ralr=True,
                 ralr_confidence=True, ralr_entropy=True, pmef_mode='prior',
                 lambda_code_init=0.1, lambda_freq_init=0.01, freeze_backbone=False,
                 freeze_residual_encoder=True, freeze_codebook=True,
                 gaussian_sigma_low=2.0, gaussian_sigma_high=1.0, return_dict=True,
                 strict_backbone_load=True, **obsolete_options):
        super().__init__()
        if not use_original_backbone:
            raise ValueError('GeoCodeSR requires the retained RRDB and Multi-scale Encoder.')
        if in_channels != 3 or out_channels != 3:
            raise ValueError('The current GeoCodeSR rebuild supports RGB input/output only.')
        if scale != 4 or gt_resolution != 256:
            raise ValueError('The current decoder is configured for x4 and 256-pixel HR patches.')
        self.scale = int(scale)
        self.token_dim = int(token_dim)
        self.use_semantic_codebook = bool(use_semantic_codebook)
        self.use_dynamic_kernel = bool(use_dynamic_kernel)
        self.use_frequency_experts = bool(use_frequency_experts)
        self.srcr_semantic_guidance = bool(srcr_semantic_guidance)
        self.srcr_retrieval = str(srcr_retrieval)
        self.use_ralr = bool(use_ralr)
        self.ralr_confidence = bool(ralr_confidence)
        self.ralr_entropy = bool(ralr_entropy)
        self.pmef_mode = 'disabled' if not self.use_frequency_experts else str(pmef_mode)
        if self.srcr_retrieval not in {'soft', 'hard'}:
            raise ValueError(f'Unsupported SRCR retrieval mode: {self.srcr_retrieval}.')
        if self.pmef_mode not in {'disabled', 'single', 'equal', 'content', 'prior'}:
            raise ValueError(f'Unsupported PMEF mode: {self.pmef_mode}.')
        self.return_dict = bool(return_dict)
        self.lambda_code = nn.Parameter(torch.tensor(float(lambda_code_init)))
        self.lambda_freq = nn.Parameter(torch.tensor(float(lambda_freq_init)))

        # These are the only two retained QuanTexSR components in Full GeoCodeSR.
        self.content_model = RRDBNet(in_channels, out_channels, scale=scale)
        channel_query = {8: 256, 16: 256, 32: 256, 64: 256, 128: 128, 256: 64, 512: 32}
        self.multiscale_encoder = MultiScaleEncoder(
            in_channels, max_depth=3, n_res_block=2, input_res=gt_resolution,
            channel_query_dict=channel_query, norm_type='gn', act_type='silu', with_attn=False)
        if backbone_ckpt:
            self.load_pretrained_backbone(backbone_ckpt, strict=strict_backbone_load)
        # QuanTexSR treats RRDB as a pretrained content prior and detaches its
        # output before the texture encoder/decoder stage.
        self.content_model.requires_grad_(False)
        if freeze_backbone:
            self.multiscale_encoder.requires_grad_(False)

        self.semantic_codebook = SemanticResidualCodebook(
            in_channels, base_channels, embed_dim, codebook_size, token_dim,
            residual_scale, temperature, hard_in_eval)
        if residual_vq_ckpt:
            self.semantic_codebook.load_residual_vq(
                residual_vq_ckpt, freeze_residual_encoder, freeze_codebook)
        elif self.use_semantic_codebook:
            raise FileNotFoundError('residual_vq_ckpt is required when semantic code retrieval is enabled.')

        # Four independent scale adapters make all spatial/channel alignment explicit.
        self.condition_align_32 = ScaleConditionAligner(embed_dim, token_dim, 256)
        self.condition_align_64 = ScaleConditionAligner(embed_dim, token_dim, 256)
        self.condition_align_128 = ScaleConditionAligner(embed_dim, token_dim, 128)
        self.condition_align_256 = ScaleConditionAligner(embed_dim, token_dim, 64)

        self.bottleneck_fusion = GatedCodeSkipFusion(256)
        self.dynamic_reassembly_32_64 = DynamicPixelReassembly(
            256, 256, 256, kernel_size, normalize_kernel=normalize_kernel)
        self.fusion_64 = GatedCodeSkipFusion(256, 256)
        self.frequency_experts_64 = SpatialRoutedFrequencyExperts(
            256, base_channels, gaussian_sigma_low, gaussian_sigma_high)

        self.dynamic_reassembly_64_128 = DynamicPixelReassembly(
            256, 128, 256, kernel_size, normalize_kernel=normalize_kernel)
        self.fusion_128 = GatedCodeSkipFusion(128, 128)
        self.frequency_experts_128 = SpatialRoutedFrequencyExperts(
            128, base_channels, gaussian_sigma_low, gaussian_sigma_high)

        self.dynamic_reassembly_128_256 = DynamicPixelReassembly(
            128, 64, 128, kernel_size, normalize_kernel=normalize_kernel)
        self.fusion_256 = GatedCodeSkipFusion(64, in_channels)
        self.frequency_experts_256 = SpatialRoutedFrequencyExperts(
            64, base_channels, gaussian_sigma_low, gaussian_sigma_high)

        self.reconstruction_head = nn.Sequential(
            QuanTexResBlock(64, 64, norm_type='gn', act_type='silu'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.content_model.eval()
        return self

    def load_pretrained_backbone(self, path, strict=True):
        checkpoint = torch.load(str(Path(path)), map_location='cpu')
        state_dict, key = extract_state_dict(checkpoint)
        state_dict = strip_prefix(state_dict, ['module.', 'backbone.'])
        targets = {
            **{f'content_model.{name}': value for name, value in self.content_model.state_dict().items()},
            **{f'multiscale_encoder.{name}': value for name, value in self.multiscale_encoder.state_dict().items()},
        }
        compatible = {name: value for name, value in state_dict.items()
                      if name in targets and targets[name].shape == value.shape}
        mismatch = sorted(name for name, value in state_dict.items()
                          if name in targets and targets[name].shape != value.shape)
        content_state = {name[len('content_model.'):]: value for name, value in compatible.items()
                         if name.startswith('content_model.')}
        encoder_state = {name[len('multiscale_encoder.'):]: value for name, value in compatible.items()
                         if name.startswith('multiscale_encoder.')}
        self.content_model.load_state_dict(content_state, strict=False)
        self.multiscale_encoder.load_state_dict(encoder_state, strict=False)
        missing = sorted(set(targets) - set(compatible) - set(mismatch))
        replaced_prefixes = (
            'quantize_group.', 'before_quant_group.', 'after_quant_group.',
            'sft_fusion_group.', 'decoder_group.', 'out_conv.')
        intentionally_replaced = sorted(
            name for name in state_dict if name.startswith(replaced_prefixes))
        unexpected = sorted(
            name for name in state_dict
            if name not in compatible and name not in mismatch and name not in intentionally_replaced)
        print(f'[GeoCodeSR] retained checkpoint key={key}: {path}')
        print(f'[GeoCodeSR] successfully loaded retained modules: {len(compatible)}/{len(targets)} tensors')
        print(f'[GeoCodeSR] intentionally replaced original modules: {len(intentionally_replaced)} tensors')
        print(f'[GeoCodeSR] retained missing={len(missing)}, shape_mismatch={len(mismatch)}, '
              f'unexpected={len(unexpected)}')
        if intentionally_replaced:
            print(f'[GeoCodeSR] replaced examples: {intentionally_replaced[:10]}')
        if missing:
            print(f'[GeoCodeSR] retained missing examples: {missing[:10]}')
        if mismatch:
            print(f'[GeoCodeSR] retained shape mismatch examples: {mismatch[:10]}')
        if unexpected:
            print(f'[GeoCodeSR] unexpected examples: {unexpected[:10]}')
        if strict and (missing or mismatch):
            raise RuntimeError(f'RRDB or Multi-scale Encoder tensors are incompatible with {path}.')
        return missing, mismatch, intentionally_replaced, unexpected

    def load_full_checkpoint(self, path, strict=False):
        checkpoint = torch.load(str(Path(path)), map_location='cpu')
        state_dict, key = extract_state_dict(checkpoint)
        state_dict = strip_prefix(state_dict, ['module.'])
        current = self.state_dict()
        compatible = {name: value for name, value in state_dict.items()
                      if name in current and current[name].shape == value.shape}
        mismatch = sorted(name for name, value in state_dict.items()
                          if name in current and current[name].shape != value.shape)
        incompatible = self.load_state_dict(compatible, strict=False)
        missing = sorted(incompatible.missing_keys)
        unexpected = sorted(set(state_dict) - set(compatible) - set(mismatch))
        print(f'[GeoCodeSR] full checkpoint key={key}: {path}')
        print(f'[GeoCodeSR] successfully loaded={len(compatible)}, missing new modules={len(missing)}, '
              f'shape_mismatch={len(mismatch)}, unexpected={len(unexpected)}')
        if missing:
            print(f'[GeoCodeSR] missing examples: {missing[:10]}')
        if mismatch:
            print(f'[GeoCodeSR] shape mismatch examples: {mismatch[:10]}')
        if unexpected:
            print(f'[GeoCodeSR] unexpected examples: {unexpected[:10]}')
        missing_required = [name for name in missing if '.single_expert.' not in name]
        if strict and (missing_required or mismatch or unexpected):
            raise RuntimeError('Full GeoCodeSR checkpoint is incompatible with the rebuilt architecture.')
        return missing, mismatch, unexpected

    def load_legacy_enhancement_checkpoint(self, path):
        raise RuntimeError(
            f'Legacy bypass-branch checkpoints cannot be migrated into the single-path decoder: {path}')

    def _empty_code_output(self, reference):
        batch, _, height, width = reference.shape
        code = reference.new_zeros(batch, self.semantic_codebook.embed_dim, height, width)
        confidence = reference.new_zeros(batch, 1, height, width)
        probability = reference.new_zeros(
            batch, self.semantic_codebook.codebook_size, height, width)
        return {
            'code_feat': code,
            'soft_code_feat': code,
            'code_indices': torch.zeros(batch, height, width, device=reference.device, dtype=torch.long),
            'pred_indices': torch.zeros(batch, height, width, device=reference.device, dtype=torch.long),
            'code_logits': probability,
            'code_prob': probability,
            'code_confidence': confidence,
            'code_entropy': confidence,
            'code_entropy_raw': confidence,
            'normalized_entropy': confidence,
            'code_uncertainty': confidence,
            'pred_perplexity': reference.new_tensor(0.0),
            'pred_codebook_usage': reference.new_tensor(0.0),
        }

    def forward(self, lq, semantic_token=None, gt_residual=None, residual_gt=None, return_dict=None):
        if lq.shape[1] != 3:
            raise ValueError(f'GeoCodeSR expects RGB input, got {tuple(lq.shape)}.')
        if gt_residual is not None and residual_gt is not None:
            raise ValueError('Pass only one of gt_residual or the legacy residual_gt alias.')
        if gt_residual is None:
            gt_residual = residual_gt
        if semantic_token is None:
            semantic_token = lq.new_zeros(lq.shape[0], self.token_dim)

        with torch.no_grad():
            content_hr = self.content_model(lq)
        texture_input = content_hr.detach()
        content_lr = F.interpolate(
            texture_input, size=lq.shape[-2:], mode='bilinear', align_corners=False)
        encoder_feats = self.multiscale_encoder(texture_input)[::-1]
        if len(encoder_feats) != 3:
            raise RuntimeError(f'Expected three encoder scales, got {len(encoder_feats)}.')
        enc_32, enc_64, enc_128 = encoder_feats

        if self.use_semantic_codebook:
            code_output = self.semantic_codebook(
                content_lr, semantic_token, gt_residual,
                semantic_guidance=self.srcr_semantic_guidance,
                retrieval_mode=self.srcr_retrieval)
        else:
            code_output = self._empty_code_output(content_lr)
        code_feat = code_output['code_feat']
        confidence = code_output['code_confidence']
        normalized_entropy = code_output['normalized_entropy']

        condition_32 = self.condition_align_32(
            code_feat, semantic_token, confidence, normalized_entropy, enc_32.shape[-2:])
        condition_64 = self.condition_align_64(
            code_feat, semantic_token, confidence, normalized_entropy, enc_64.shape[-2:])
        condition_128 = self.condition_align_128(
            code_feat, semantic_token, confidence, normalized_entropy, enc_128.shape[-2:])
        condition_256 = self.condition_align_256(
            code_feat, semantic_token, confidence, normalized_entropy, texture_input.shape[-2:])
        aligned_conditions = [condition_32, condition_64, condition_128, condition_256]
        lambda_code = self.lambda_code
        lambda_freq = self.lambda_freq

        feature_32, code_delta_32, fusion_32 = self.bottleneck_fusion(
            enc_32, None, condition_32, lambda_code, enabled=self.use_ralr,
            use_confidence=self.ralr_confidence, use_entropy=self.ralr_entropy)

        feature_64, kernel_64, kernel_entropy_64, kernel_smoothness_64 = \
            self.dynamic_reassembly_32_64(
                feature_32, condition_32, self.use_dynamic_kernel and self.use_ralr,
                self.ralr_confidence, self.ralr_entropy)
        feature_64, code_delta_64, fusion_64 = self.fusion_64(
            feature_64, enc_64, condition_64, lambda_code, enabled=self.use_ralr,
            use_confidence=self.ralr_confidence, use_entropy=self.ralr_entropy)
        pmef_mode = self.pmef_mode if self.use_frequency_experts else 'disabled'
        feature_64, frequency_64 = self.frequency_experts_64(
            feature_64, condition_64, lambda_freq, pmef_mode)

        feature_128, kernel_128, kernel_entropy_128, kernel_smoothness_128 = \
            self.dynamic_reassembly_64_128(
                feature_64, condition_64, self.use_dynamic_kernel and self.use_ralr,
                self.ralr_confidence, self.ralr_entropy)
        feature_128, code_delta_128, fusion_128 = self.fusion_128(
            feature_128, enc_128, condition_128, lambda_code, enabled=self.use_ralr,
            use_confidence=self.ralr_confidence, use_entropy=self.ralr_entropy)
        feature_128, frequency_128 = self.frequency_experts_128(
            feature_128, condition_128, lambda_freq, pmef_mode)

        feature_256, kernel_256, kernel_entropy_256, kernel_smoothness_256 = \
            self.dynamic_reassembly_128_256(
                feature_128, condition_128, self.use_dynamic_kernel and self.use_ralr,
                self.ralr_confidence, self.ralr_entropy)
        feature_256, code_delta_256, fusion_256 = self.fusion_256(
            feature_256, texture_input, condition_256, lambda_code, enabled=self.use_ralr,
            use_confidence=self.ralr_confidence, use_entropy=self.ralr_entropy)
        feature_256, frequency_256 = self.frequency_experts_256(
            feature_256, condition_256, lambda_freq, pmef_mode)

        sr_final = self.reconstruction_head(feature_256)
        sr_low = gaussian_blur(sr_final.float(), self.frequency_experts_256.sigma_low)
        sr_high_blur = gaussian_blur(sr_final.float(), self.frequency_experts_256.sigma_high)
        frequency_band_predictions = [
            sr_low, sr_high_blur - sr_low, sr_final.float() - sr_high_blur]
        freq_image_delta = frequency_band_predictions[1] + frequency_band_predictions[2]

        kernels = [kernel_64, kernel_128, kernel_256]
        frequency_outputs = [frequency_64, frequency_128, frequency_256]
        routing_weights_all = [item['weights'] for item in frequency_outputs]
        expert_loads = torch.stack([item['load'] for item in frequency_outputs], dim=0)
        expert_hard_loads = torch.stack([
            item['hard_load'] for item in frequency_outputs], dim=0)
        expert_norms = torch.stack([item['expert_norms'] for item in frequency_outputs], dim=0)
        routing_entropies = torch.stack([item['entropy'] for item in frequency_outputs])
        kernel_entropies = torch.stack([
            kernel_entropy_64, kernel_entropy_128, kernel_entropy_256])
        kernel_smoothness = torch.stack([
            kernel_smoothness_64, kernel_smoothness_128, kernel_smoothness_256])
        balance_loss = torch.stack([item['balance'] for item in frequency_outputs]).mean()
        router_z_loss = torch.stack([
            item['router_z_loss'] for item in frequency_outputs]).mean()
        fusion_stats = {
            '32': fusion_32,
            '64': fusion_64,
            '128': fusion_128,
            '256': fusion_256,
        }

        output = {
            'sr': sr_final,
            'sr_final': sr_final,
            'content_feat': content_hr,
            'content_hr': content_hr,
            'encoder_feats': encoder_feats,
            'decoder_features': [feature_64, feature_128, feature_256],
            'final_feat': feature_256,
            'code_feat': code_feat,
            'code_indices': code_output['code_indices'],
            'pred_indices': code_output['pred_indices'],
            'code_logits': code_output['code_logits'],
            'code_prob': code_output['code_prob'],
            'code_confidence': confidence,
            'code_entropy': code_output['code_entropy'],
            'code_entropy_raw': code_output['code_entropy_raw'],
            'normalized_entropy': normalized_entropy,
            'code_uncertainty': normalized_entropy,
            'pred_perplexity': code_output['pred_perplexity'],
            'pred_codebook_usage': code_output['pred_codebook_usage'],
            'aligned_conditions': aligned_conditions,
            'code_aligned': [condition['code'] for condition in aligned_conditions],
            'semantic_aligned': [condition['semantic'] for condition in aligned_conditions],
            'confidence_aligned': [condition['confidence'] for condition in aligned_conditions],
            'entropy_aligned': [condition['entropy'] for condition in aligned_conditions],
            'code_deltas': [code_delta_32, code_delta_64, code_delta_128, code_delta_256],
            'fusion_gates': fusion_stats,
            'code_gates': [fusion_32['code_gate'], fusion_64['code_gate'],
                           fusion_128['code_gate'], fusion_256['code_gate']],
            'reassembly_kernels': kernels,
            'dynamic_kernels': kernels,
            'kernel_entropy': kernel_entropies.mean(),
            'kernel_entropies': kernel_entropies,
            'kernel_smoothness': kernel_smoothness.mean(),
            'kernel_smoothness_all': kernel_smoothness,
            'frequency_deltas': [item['delta'] for item in frequency_outputs],
            'frequency_expert_outputs': [item['expert_outputs'] for item in frequency_outputs],
            'frequency_band_predictions': frequency_band_predictions,
            'freq_delta': freq_image_delta,
            'freq_image_delta': freq_image_delta,
            'routing_weights': frequency_256['weights'],
            'routing_weights_all': routing_weights_all,
            'routing_entropy': routing_entropies.mean(),
            'spatial_routing_entropy': routing_entropies,
            'expert_load': expert_loads.mean(dim=0),
            'expert_loads': expert_loads,
            'expert_hard_load': expert_hard_loads.mean(dim=0),
            'expert_hard_loads': expert_hard_loads,
            'expert_norms': expert_norms,
            'expert_collapse_scores': torch.stack([
                item['collapse_score'] for item in frequency_outputs]),
            'lambda_code': lambda_code,
            'lambda_freq': lambda_freq,
            'lambda_code_raw': self.lambda_code,
            'lambda_freq_raw': self.lambda_freq,
            'aux_losses': {
                'expert_balance': balance_loss,
                'router_z_loss': router_z_loss,
            },
        }
        for key in ['target_indices', 'target_perplexity', 'target_codebook_usage', 'code_acc']:
            if key in code_output:
                output[key] = code_output[key]
        use_dict = self.return_dict if return_dict is None else bool(return_dict)
        return output if use_dict else sr_final
