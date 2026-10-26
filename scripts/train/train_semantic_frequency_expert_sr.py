import argparse
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from basicsr.archs.gf_semantic_frequency_expert_sr_arch import (
    GFSemanticFrequencyExpertSRNet, gaussian_blur)
from basicsr.data.gf_paired_semantic_dataset import GFPairedSemanticDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train the single-path rebuilt GeoCodeSR.')
    parser.add_argument('-opt', required=True)
    return parser.parse_args()


def build_loader(opt):
    dataset = GFPairedSemanticDataset(opt)
    return DataLoader(
        dataset, batch_size=opt.get('batch_size', 4), shuffle=opt.get('shuffle', True),
        num_workers=opt.get('num_workers', 0), drop_last=opt.get('drop_last', True),
        pin_memory=opt.get('pin_memory', False))


def cycle_loader(loader):
    while True:
        yield from loader


def to_device(batch, device):
    return {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def fft_loss(prediction, target):
    pred_fft = torch.abs(torch.fft.rfft2(prediction.float(), dim=(-2, -1), norm='ortho'))
    target_fft = torch.abs(torch.fft.rfft2(target.float(), dim=(-2, -1), norm='ortho'))
    return F.l1_loss(pred_fft, target_fft)


def sam_proxy_loss(prediction, target):
    pred = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1]).float()
    gt = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1]).float()
    cosine = torch.sum(pred * gt, dim=1) / (torch.norm(pred, dim=1) * torch.norm(gt, dim=1) + 1e-8)
    return torch.mean(1.0 - torch.clamp(cosine, -1.0, 1.0))


def tensor_psnr(prediction, target):
    rmse = torch.sqrt(torch.mean((prediction.float() - target.float()) ** 2)).item()
    return float('inf') if rmse < 1e-12 else 20.0 * math.log10(1.0 / rmse)


def tensor_sam(prediction, target):
    pred = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1]).float()
    gt = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1]).float()
    cosine = torch.sum(pred * gt, dim=1) / (torch.norm(pred, dim=1) * torch.norm(gt, dim=1) + 1e-8)
    return torch.mean(torch.acos(torch.clamp(cosine, -1.0, 1.0))) * 180.0 / math.pi


def zero_loss(reference):
    return reference.new_zeros(())


def frequency_band_targets(target, sigma_low=2.0, sigma_high=1.0):
    low = gaussian_blur(target.float(), sigma=sigma_low)
    high_blur = gaussian_blur(target.float(), sigma=sigma_high)
    return low, high_blur - low, target.float() - high_blur


def calc_losses(model, batch, opt, perceptual_loss=None):
    output = model(batch['lq'], batch['semantic_token'], gt_residual=batch.get('residual'))
    sr, gt = output['sr'], batch['gt']
    loss_reconstruction = F.l1_loss(sr, gt)
    loss_fft = fft_loss(sr, gt) if opt.get('fft_weight', 0.0) > 0 else zero_loss(sr)
    loss_sam = sam_proxy_loss(sr, gt) if opt.get('sam_weight', 0.0) > 0 else zero_loss(sr)
    if 'target_indices' in output and model.use_semantic_codebook:
        loss_codebook = F.cross_entropy(output['code_logits'], output['target_indices'])
    else:
        loss_codebook = zero_loss(sr)
    frequency_enabled = bool(
        opt.get('frequency_supervision_enabled', True)
        and opt.get('frequency_weight', 0.0) > 0
        and model.use_frequency_experts)
    if frequency_enabled:
        targets = frequency_band_targets(
            gt, opt.get('gaussian_sigma_low', 2.0), opt.get('gaussian_sigma_high', 1.0))
        predictions = output['frequency_band_predictions']
        band_weights = [
            float(opt.get('frequency_low_weight', 1.0)),
            float(opt.get('frequency_mid_weight', 1.0)),
            float(opt.get('frequency_high_weight', 1.0)),
        ]
        frequency_losses = [
            F.l1_loss(prediction.float(), target.float())
            for prediction, target in zip(predictions, targets)]
        weight_sum = max(sum(band_weights), 1e-12)
        loss_frequency = sum(
            weight * loss for weight, loss in zip(band_weights, frequency_losses)) / weight_sum
        loss_frequency_low, loss_frequency_mid, loss_frequency_high = frequency_losses
    else:
        loss_frequency = zero_loss(sr)
        loss_frequency_low = zero_loss(sr)
        loss_frequency_mid = zero_loss(sr)
        loss_frequency_high = zero_loss(sr)
    loss_balance = output['aux_losses']['expert_balance']
    loss_router_z = output['aux_losses']['router_z_loss']
    loss_perceptual = zero_loss(sr)
    loss_style = zero_loss(sr)
    if perceptual_loss is not None:
        loss_perceptual, loss_style = perceptual_loss(sr.float(), gt.float())
        loss_perceptual = loss_perceptual if loss_perceptual is not None else zero_loss(sr)
        loss_style = loss_style if loss_style is not None else zero_loss(sr)

    loss_total = (
        opt.get('reconstruction_weight', 1.0) * loss_reconstruction
        + opt.get('perceptual_weight', 0.0) * loss_perceptual
        + opt.get('style_weight', 0.0) * loss_style
        + opt.get('codebook_weight', 1.0) * loss_codebook
        + opt.get('fft_weight', 0.0) * loss_fft
        + opt.get('sam_weight', 0.0) * loss_sam
        + opt.get('frequency_weight', 0.0) * loss_frequency
        + opt.get('balance_weight', 0.0) * loss_balance
        + opt.get('router_z_weight', 0.0) * loss_router_z)
    if not torch.isfinite(loss_total):
        raise FloatingPointError(
            'Full GeoCodeSR total loss is NaN/Inf. Check expert norms, routing entropy, '
            'kernel entropy, and the current learning rate.')
    losses = {
        'loss_total': loss_total,
        'loss_reconstruction': loss_reconstruction,
        'loss_perceptual': loss_perceptual,
        'loss_style': loss_style,
        'loss_codebook': loss_codebook,
        'loss_fft': loss_fft,
        'loss_sam': loss_sam,
        'loss_frequency': loss_frequency,
        'loss_frequency_low': loss_frequency_low,
        'loss_frequency_mid': loss_frequency_mid,
        'loss_frequency_high': loss_frequency_high,
        'loss_balance': loss_balance,
        'loss_router_z': loss_router_z,
        'lambda_code': output['lambda_code'],
        'lambda_freq': output['lambda_freq'],
        'codebook_usage': output['pred_codebook_usage'],
        'routing_entropy': output['routing_entropy'],
        'kernel_entropy': output['kernel_entropy'],
        'expert_load_low': output['expert_load'][0],
        'expert_load_mid': output['expert_load'][1],
        'expert_load_high': output['expert_load'][2],
        'expert_hard_load_low': output['expert_hard_load'][0],
        'expert_hard_load_mid': output['expert_hard_load'][1],
        'expert_hard_load_high': output['expert_hard_load'][2],
        'expert_norm_low': output['expert_norms'][:, 0].mean(),
        'expert_norm_mid': output['expert_norms'][:, 1].mean(),
        'expert_norm_high': output['expert_norms'][:, 2].mean(),
        'expert_collapse_score': output['expert_collapse_scores'].max(),
        'fusion_skip_gate_64': output['fusion_gates']['64']['skip_gate_mean'],
        'fusion_code_gate_64': output['fusion_gates']['64']['code_gate_mean'],
        'fusion_skip_gate_128': output['fusion_gates']['128']['skip_gate_mean'],
        'fusion_code_gate_128': output['fusion_gates']['128']['code_gate_mean'],
        'fusion_skip_gate_256': output['fusion_gates']['256']['skip_gate_mean'],
        'fusion_code_gate_256': output['fusion_gates']['256']['code_gate_mean'],
        'code_confidence_mean': output['code_confidence'].mean(),
        'code_entropy_mean': output['normalized_entropy'].mean(),
    }
    return output, losses


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums, count = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch['lq'], batch['semantic_token'])
        metrics = {
            'val_psnr': tensor_psnr(output['sr'], batch['gt']),
            'val_mae': float(F.l1_loss(output['sr'], batch['gt'])),
            'val_sam': float(tensor_sam(output['sr'], batch['gt'])),
            'val_codebook_usage': float(output['pred_codebook_usage']),
            'val_routing_entropy': float(output['routing_entropy']),
            'val_kernel_entropy': float(output['kernel_entropy']),
            'val_expert_collapse_score': float(output['expert_collapse_scores'].max()),
            'val_code_confidence': float(output['code_confidence'].mean()),
            'val_code_entropy': float(output['normalized_entropy'].mean()),
        }
        if 'code_acc' in output:
            metrics['val_code_acc'] = float(output['code_acc'])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def rgb_uint8(tensor):
    image = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def save_preview(batch, output, path):
    error = torch.abs(output['sr'][0] - batch['gt'][0])
    panels = [rgb_uint8(batch['lq'][0]), rgb_uint8(output['sr'][0]),
              rgb_uint8(batch['gt'][0]), rgb_uint8(error)]
    target_height = panels[1].shape[0]
    panels[0] = cv2.resize(panels[0], (panels[1].shape[1], target_height), interpolation=cv2.INTER_CUBIC)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))


def build_optimizer(model, train_opt):
    base_lr = float(train_opt.get('lr', 2e-4))
    backbone_lr = base_lr * float(train_opt.get('backbone_lr_mult', 0.1))
    backbone_params, enhancement_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        retained_backbone = name.startswith(('content_model.', 'multiscale_encoder.'))
        (backbone_params if retained_backbone else enhancement_params).append(parameter)
    groups = []
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'})
    if enhancement_params:
        groups.append({'params': enhancement_params, 'lr': base_lr, 'name': 'enhancement'})
    print(f'Optimizer groups: backbone={len(backbone_params)} lr={backbone_lr:.3e}, '
          f'enhancement={len(enhancement_params)} lr={base_lr:.3e}')
    betas = tuple(float(value) for value in train_opt.get('betas', [0.9, 0.99]))
    return torch.optim.Adam(
        groups, lr=base_lr, betas=betas,
        weight_decay=float(train_opt.get('weight_decay', 0.0)))


def build_perceptual_loss(train_opt, device):
    if train_opt.get('perceptual_weight', 0.0) <= 0 and train_opt.get('style_weight', 0.0) <= 0:
        return None
    from basicsr.losses.losses import PerceptualLoss
    return PerceptualLoss(
        layer_weights=train_opt.get('perceptual_layers', {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1.0}),
        vgg_type='vgg19', use_input_norm=True, range_norm=False,
        perceptual_weight=1.0, style_weight=1.0, criterion='l1').to(device)


def save_checkpoint(model, optimizer, scaler, cfg, current_iter, save_dir, latest=False,
                    filename=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = ('net_geocodesr_latest.pth' if latest
                    else f'net_geocodesr_iter_{current_iter:06d}.pth')
    torch.save({
        'iter': current_iter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': cfg,
    }, save_dir / filename)


def main():
    args = parse_args()
    with open(args.opt, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    seed = int(cfg.get('manual_seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('num_gpu', 1) > 0 else 'cpu')
    train_loader = build_loader(cfg['datasets']['train'])
    val_loader = build_loader(cfg['datasets']['val'])
    train_iterator = cycle_loader(train_loader)
    network_opt = cfg['network'].copy()
    network_opt.pop('type', None)
    model = GFSemanticFrequencyExpertSRNet(**network_opt).to(device)
    train_opt = cfg['train']
    if cfg.get('path', {}).get('pretrain_full'):
        model.load_full_checkpoint(cfg['path']['pretrain_full'], strict=False)
    if cfg.get('path', {}).get('legacy_enhancement'):
        model.load_legacy_enhancement_checkpoint(cfg['path']['legacy_enhancement'])
    optimizer = build_optimizer(model, train_opt)
    perceptual_loss = build_perceptual_loss(train_opt, device)
    amp_enabled = bool(train_opt.get('amp', True) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    experiment_root = Path(cfg['path']['experiments_root'])
    model_dir = experiment_root / 'models'
    visualization_dir = experiment_root / 'visualization'
    total_iter = int(train_opt.get('total_iter', 30000))
    metric_history = []
    best_val_psnr = -float('inf')
    print(f'Device: {device}, AMP: {amp_enabled}, total_iter: {total_iter}')
    print(f'Experiment root: {experiment_root}')
    model.train()
    for current_iter in range(1, total_iter + 1):
        batch = to_device(next(train_iterator), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output, losses = calc_losses(model, batch, train_opt, perceptual_loss)
        scaler.scale(losses['loss_total']).backward()
        scaler.unscale_(optimizer)
        grad_clip_norm = float(train_opt.get('grad_clip_norm', 0.0))
        trainable_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None]
        if grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, grad_clip_norm, error_if_nonfinite=False)
        else:
            grad_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(parameter.grad.detach().float())
                for parameter in trainable_parameters]))
        scaler.step(optimizer)
        scaler.update()
        losses['grad_norm'] = grad_norm
        if current_iter == 1 or current_iter % int(train_opt.get('print_freq', 20)) == 0:
            print(f'iter={current_iter} ' + ' '.join(
                f'{key}={float(value):.6f}' for key, value in losses.items()))
        if current_iter % int(train_opt.get('val_freq', 1000)) == 0:
            metrics = validate(model, val_loader, device)
            print(f'val iter={current_iter} ' + ' '.join(f'{key}={value:.6f}' for key, value in metrics.items()))
            metric_history.append({'iter': current_iter, **metrics})
            (experiment_root / 'metrics.json').write_text(
                json.dumps(metric_history, indent=2, allow_nan=False), encoding='utf-8')
            if metrics['val_psnr'] > best_val_psnr:
                best_val_psnr = metrics['val_psnr']
                save_checkpoint(model, optimizer, scaler, cfg, current_iter, model_dir,
                                filename='best.pth')
        if current_iter % int(train_opt.get('save_preview_freq', 1000)) == 0:
            save_preview(batch, output, visualization_dir / f'preview_{current_iter:06d}.png')
        if current_iter % int(train_opt.get('save_checkpoint_freq', 2000)) == 0:
            save_checkpoint(model, optimizer, scaler, cfg, current_iter, model_dir)
            save_checkpoint(model, optimizer, scaler, cfg, current_iter, model_dir, latest=True)
    save_checkpoint(model, optimizer, scaler, cfg, total_iter, model_dir, latest=True)
    save_checkpoint(model, optimizer, scaler, cfg, total_iter, model_dir, filename='last.pth')
    print('Training done.')


if __name__ == '__main__':
    main()
