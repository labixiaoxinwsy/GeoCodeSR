import argparse
import importlib.util
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import rasterio
import torch
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_class(file_path, class_name):
    spec = importlib.util.spec_from_file_location(Path(file_path).stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


GFResidualVQVAE = load_class(
    ROOT_DIR / 'basicsr' / 'archs' / 'gf_residual_vqvae_arch.py',
    'GFResidualVQVAE')


def parse_args():
    parser = argparse.ArgumentParser(description='Infer residual VQ codebook reconstructions.')
    parser.add_argument('--ckpt', required=True, help='Residual VQ checkpoint.')
    parser.add_argument('--dataroot_residual', default='E:/GeoCodeSR/tuihua_paired_x4/test/Residual_x4')
    parser.add_argument('--output_dir', default='E:/GeoCodeSR/tuihua_paired_x4/results/residual_vq_recon')
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--residual_scale', type=float, default=20.0)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--save_preview', action='store_true')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def list_tifs(directory):
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}])


def read_residual(path, bands):
    with rasterio.open(path) as src:
        data = src.read(list(range(1, bands + 1)))
        profile = src.profile.copy()
    image = np.transpose(data, (1, 2, 0)).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f'NaN/Inf in residual: {path}')
    return image, profile


def save_tif(path, image, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.astype(np.float32)
    profile = profile.copy()
    profile.update(driver='GTiff', height=image.shape[0], width=image.shape[1], count=image.shape[2], dtype='float32')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.transpose(image, (2, 0, 1)))


def minmax_uint8(image):
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)


def rgb_preview(image):
    return minmax_uint8(image[:, :, [2, 1, 0]])


def band_preview(image, band=-1):
    band = min(max(int(band), 0), image.shape[2] - 1)
    return cv2.cvtColor(minmax_uint8(image[:, :, band]), cv2.COLOR_GRAY2RGB)


def save_preview(path, residual, recon):
    error = np.abs(recon - residual)
    panels = [
        rgb_preview(residual),
        rgb_preview(recon),
        rgb_preview(error),
        band_preview(residual, residual.shape[2] - 1),
        band_preview(recon, recon.shape[2] - 1),
    ]
    canvas = np.concatenate(panels, axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def fft_mae(pred, target):
    pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).float()
    target_t = torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0).float()
    pred_fft = torch.fft.rfft2(pred_t, dim=(-2, -1), norm='ortho')
    target_fft = torch.fft.rfft2(target_t, dim=(-2, -1), norm='ortho')
    return float(F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft)))


def calc_metrics(recon, target):
    diff = recon - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = float('inf') if rmse < 1e-12 else float(20 * math.log10(1.0 / rmse))
    return {
        'mae': mae,
        'rmse': rmse,
        'psnr': psnr,
        'fft_mae': fft_mae(recon, target),
    }


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt.get('config', {})
    net_opt = cfg.get('network', {
        'in_channels': 4,
        'out_channels': 4,
        'base_channels': 64,
        'embed_dim': 64,
        'codebook_size': 512,
        'commitment_weight': 0.25,
    })
    model = GFResidualVQVAE(**net_opt).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    residual_scale = float(ckpt.get('residual_scale', args.residual_scale))
    model = build_model_from_ckpt(ckpt, device)

    paths = list_tifs(args.dataroot_residual)
    if args.num_samples is not None and args.num_samples > 0:
        paths = paths[:args.num_samples]
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / 'preview'

    print(f'Checkpoint: {args.ckpt}')
    print(f'Device: {device}')
    print(f'Residual scale: {residual_scale}')
    print(f'Samples: {len(paths)}')

    metrics_list = []
    perplexities = []
    usages = []
    for path in paths:
        residual, profile = read_residual(path, args.bands)
        inp = torch.from_numpy((residual * residual_scale).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(inp)
        recon_scaled = output['recon'].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        recon = recon_scaled / residual_scale
        save_tif(output_dir / path.name, recon, profile)
        if args.save_preview:
            save_preview(preview_dir / f'{path.stem}_preview.png', residual, recon)

        metrics = calc_metrics(recon, residual)
        metrics_list.append(metrics)
        perplexities.append(float(output['perplexity']))
        usages.append(float(output['codebook_usage']))
        print(
            f'{path.name}: MAE={metrics["mae"]:.6f}, RMSE={metrics["rmse"]:.6f}, '
            f'PSNR={metrics["psnr"]:.6f}, FFT_MAE={metrics["fft_mae"]:.6f}, '
            f'perplexity={perplexities[-1]:.4f}, usage={usages[-1]:.4f}')

    if metrics_list:
        print('\nAverage metrics on original residual range:')
        for key in ['mae', 'rmse', 'psnr', 'fft_mae']:
            print(f'  {key.upper()}: {np.mean([m[key] for m in metrics_list]):.6f}')
        print(f'  PERPLEXITY: {np.mean(perplexities):.4f}')
        print(f'  CODEBOOK_USAGE: {np.mean(usages):.4f}')
    print('Done.')


if __name__ == '__main__':
    main()
