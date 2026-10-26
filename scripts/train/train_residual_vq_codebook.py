import argparse
import importlib.util
import random
from pathlib import Path
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


def load_class(file_path, class_name):
    spec = importlib.util.spec_from_file_location(Path(file_path).stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


GFResidualCodebookDataset = load_class(
    ROOT_DIR / 'basicsr' / 'data' / 'gf_residual_codebook_dataset.py',
    'GFResidualCodebookDataset')
GFResidualVQVAE = load_class(
    ROOT_DIR / 'basicsr' / 'archs' / 'gf_residual_vqvae_arch.py',
    'GFResidualVQVAE')


def parse_args():
    parser = argparse.ArgumentParser(description='Train residual VQ codebook.')
    parser.add_argument('-opt', required=True, help='Path to YAML config.')
    return parser.parse_args()


def build_loader(dataset_opt):
    dataset = GFResidualCodebookDataset(dataset_opt)
    return DataLoader(
        dataset,
        batch_size=dataset_opt.get('batch_size', 4),
        shuffle=dataset_opt.get('shuffle', True),
        num_workers=dataset_opt.get('num_workers', 0),
        drop_last=dataset_opt.get('drop_last', True),
        pin_memory=dataset_opt.get('pin_memory', False))


def fft_magnitude_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
    target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def minmax_uint8(image):
    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def rgb_preview(tensor):
    image = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    if image.shape[2] < 3:
        return None
    return minmax_uint8(image[:, :, [2, 1, 0]])


def band_preview(tensor, band=3):
    band = min(max(int(band), 0), tensor.shape[0] - 1)
    image = tensor.detach().cpu().float().numpy()[band]
    return cv2.cvtColor(minmax_uint8(image), cv2.COLOR_GRAY2RGB)


def save_preview(batch, output, save_path, residual_scale):
    target = batch['residual_scaled'][:1] / residual_scale
    recon = output['recon'][:1] / residual_scale
    error = torch.abs(recon - target)
    panels = [
        rgb_preview(target[0]),
        rgb_preview(recon[0]),
        rgb_preview(error[0]),
        band_preview(target[0], 2),
        band_preview(recon[0], 2),
    ]
    if any(panel is None for panel in panels):
        return
    canvas = np.concatenate(panels, axis=1)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def move_batch_to_device(batch, device):
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def calc_losses(model, batch, train_opt):
    target = batch['residual_scaled']
    output = model(target)
    recon = output['recon']
    loss_l1 = F.l1_loss(recon, target)
    loss_l2 = F.mse_loss(recon, target)
    loss_fft = fft_magnitude_loss(recon, target)
    loss_total = (
        train_opt.get('l1_weight', 1.0) * loss_l1
        + train_opt.get('l2_weight', 0.0) * loss_l2
        + train_opt.get('fft_weight', 0.05) * loss_fft
        + train_opt.get('vq_loss_weight', 1.0) * output['vq_loss']
    )
    finite_tensors = {
        'target': target,
        'reconstruction': recon,
        'vq_loss': output['vq_loss'],
        'perplexity': output['perplexity'],
        'code_probabilities': output['code_probabilities'],
    }
    invalid = [name for name, value in finite_tensors.items() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(f'Stage 1 NaN/Inf check failed for: {invalid}')
    losses = {
        'loss_total': loss_total,
        'loss_l1': loss_l1,
        'loss_l2': loss_l2,
        'loss_fft': loss_fft,
        'vq_loss': output['vq_loss'],
        'commitment_loss': output['commitment_loss'],
        'perplexity': output['perplexity'],
        'codebook_usage': output['codebook_usage'],
        'dead_code_ratio': output['dead_code_ratio'],
    }
    return output, losses


@torch.no_grad()
def validate(model, val_loader, device, train_opt):
    model.eval()
    sums = {'val_l1': 0.0, 'val_fft': 0.0, 'val_perplexity': 0.0,
            'val_codebook_usage': 0.0, 'val_dead_code_ratio': 0.0}
    count = 0
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        output, losses = calc_losses(model, batch, train_opt)
        sums['val_l1'] += float(losses['loss_l1'])
        sums['val_fft'] += float(losses['loss_fft'])
        sums['val_perplexity'] += float(losses['perplexity'])
        sums['val_codebook_usage'] += float(losses['codebook_usage'])
        sums['val_dead_code_ratio'] += float(losses['dead_code_ratio'])
        count += 1
    model.train()
    if count == 0:
        return sums
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(model, optimizer, cfg, current_iter, residual_scale, save_dir, latest=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = 'net_residual_vq_latest.pth' if latest else f'net_residual_vq_iter_{current_iter:06d}.pth'
    torch.save({
        'iter': current_iter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': cfg,
        'residual_scale': residual_scale,
    }, save_dir / name)


def main():
    args = parse_args()
    with open(args.opt, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get('manual_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('num_gpu', 1) > 0 else 'cpu')
    exp_root = Path(cfg['path']['experiments_root'])
    model_dir = exp_root / 'models'
    vis_dir = exp_root / 'visualization'
    model_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg['datasets']['train'])
    val_loader = build_loader(cfg['datasets']['val']) if 'val' in cfg.get('datasets', {}) else None
    train_iter = cycle_loader(train_loader)

    model = GFResidualVQVAE(**cfg['network']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train'].get('lr', 2e-4),
        weight_decay=cfg['train'].get('weight_decay', 0.0))
    train_opt = cfg['train']
    total_iter = train_opt.get('total_iter', 5000)
    residual_scale = cfg['datasets']['train'].get('residual_scale', 20.0)

    print(f'Device: {device}')
    print(f'Total iter: {total_iter}')
    print(f'Experiment root: {exp_root}')

    model.train()
    for current_iter in range(1, total_iter + 1):
        batch = move_batch_to_device(next(train_iter), device)
        output, losses = calc_losses(model, batch, train_opt)
        optimizer.zero_grad()
        losses['loss_total'].backward()
        optimizer.step()

        if current_iter % train_opt.get('print_freq', 20) == 0 or current_iter == 1:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"iter={current_iter} lr={lr:.3e} "
                f"loss_total={float(losses['loss_total']):.6f} "
                f"loss_l1={float(losses['loss_l1']):.6f} "
                f"loss_fft={float(losses['loss_fft']):.6f} "
                f"vq_loss={float(losses['vq_loss']):.6f} "
                f"commitment_loss={float(losses['commitment_loss']):.6f} "
                f"perplexity={float(losses['perplexity']):.4f} "
                f"codebook_usage={float(losses['codebook_usage']):.4f} "
                f"dead_code_ratio={float(losses['dead_code_ratio']):.4f}")
            probabilities = output['code_probabilities'].detach()
            top_probability, top_index = torch.topk(probabilities, k=min(10, probabilities.numel()))
            distribution = ', '.join(
                f'{int(index)}:{float(probability):.6f}'
                for index, probability in zip(top_index, top_probability))
            print(f'  top_code_distribution={distribution}')

        if val_loader is not None and current_iter % train_opt.get('val_freq', 500) == 0:
            val_metrics = validate(model, val_loader, device, train_opt)
            print(
                f"val iter={current_iter} val_l1={val_metrics['val_l1']:.6f} "
                f"val_fft={val_metrics['val_fft']:.6f} "
                f"val_perplexity={val_metrics['val_perplexity']:.4f} "
                f"val_codebook_usage={val_metrics['val_codebook_usage']:.4f} "
                f"val_dead_code_ratio={val_metrics['val_dead_code_ratio']:.4f}")

        if current_iter % train_opt.get('save_preview_freq', 500) == 0:
            save_preview(batch, output, vis_dir / f'preview_iter_{current_iter:06d}.png', residual_scale)

        if current_iter % train_opt.get('save_checkpoint_freq', 1000) == 0:
            save_checkpoint(model, optimizer, cfg, current_iter, residual_scale, model_dir, latest=False)
            save_checkpoint(model, optimizer, cfg, current_iter, residual_scale, model_dir, latest=True)

    save_checkpoint(model, optimizer, cfg, total_iter, residual_scale, model_dir, latest=True)
    print('Training done.')


if __name__ == '__main__':
    main()
