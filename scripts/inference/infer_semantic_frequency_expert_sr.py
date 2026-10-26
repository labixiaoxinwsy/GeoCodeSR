import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import rasterio
from affine import Affine
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from basicsr.archs.gf_semantic_frequency_expert_sr_arch import (
    GFSemanticFrequencyExpertSRNet as GeoCodeSR)


def parse_args():
    parser = argparse.ArgumentParser(description='Infer the single-path rebuilt GeoCodeSR.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataroot_lq', default='E:/GeoCodeSR/tuihua_paired_x4/test/LR_x4')
    parser.add_argument('--dataroot_gt', default='E:/GeoCodeSR/tuihua_paired_x4/test/HR')
    parser.add_argument('--dataroot_semantic', default='E:/GeoCodeSR/tuihua_paired_x4/test/SatMAE_tokens_mean')
    parser.add_argument('--output_dir', default='E:/GeoCodeSR/tuihua_paired_x4/results/geocodesr')
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument(
        '--input_range', type=float, default=None,
        help='Radiometric value represented by model value 1.0. Auto: 255 for uint8, 10000 for other integer TIFFs.')
    parser.add_argument(
        '--output_dtype', default='source', choices=['source', 'float32'],
        help='Write the source TIFF dtype/range, or normalized float32 values in [0, 1].')
    parser.add_argument('--baseline_only', action='store_true')
    parser.add_argument('--disable_semantic_codebook', action='store_true')
    parser.add_argument('--disable_dynamic_kernel', action='store_true')
    parser.add_argument('--disable_frequency_experts', action='store_true')
    parser.add_argument('--save_preview', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip outputs that already exist.')
    parser.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision for faster inference.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for equal-sized TIFF tiles.')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def list_tifs(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f'Directory does not exist: {directory}')
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in {'.tif', '.tiff'})
    if not paths:
        raise FileNotFoundError(f'No tif/tiff files found in: {directory}')
    return paths


def infer_input_range(dtypes, configured_range=None):
    if configured_range is not None:
        if configured_range <= 0:
            raise ValueError('--input_range must be greater than zero.')
        return float(configured_range)
    dtypes = [np.dtype(dtype) for dtype in dtypes]
    if all(dtype == np.dtype('uint8') for dtype in dtypes):
        return 255.0
    if all(np.issubdtype(dtype, np.integer) for dtype in dtypes):
        # The training pipeline uses reflectance scaled to 0..10000 for non-uint8 data.
        return 10000.0
    return 1.0


def read_tif(path, bands, input_range=None, clip=True):
    with rasterio.open(path) as source:
        if source.count != bands:
            raise ValueError(f'{path} has {source.count} bands; GeoCodeSR requires exactly {bands}.')
        data = source.read(list(range(1, bands + 1)))
        profile = source.profile.copy()
        metadata = {
            'profile': profile,
            'dtype': np.dtype(source.dtypes[0]),
            'input_range': infer_input_range(source.dtypes[:bands], input_range),
            'dataset_tags': source.tags(),
            'band_tags': [source.tags(index) for index in range(1, bands + 1)],
            'descriptions': source.descriptions,
            'units': source.units,
            'colorinterp': source.colorinterp,
            'scales': source.scales,
            'offsets': source.offsets,
        }
    image = np.transpose(data, (1, 2, 0)).astype(np.float32) / metadata['input_range']
    if not np.isfinite(image).all():
        raise ValueError(f'NaN/Inf found in input TIFF: {path}')
    return (np.clip(image, 0.0, 1.0) if clip else image).astype(np.float32), metadata


def read_token(path):
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        value = value.item()
    token = value['token'] if isinstance(value, dict) else value
    token = np.asarray(token, dtype=np.float32)
    if not np.isfinite(token).all():
        raise ValueError(f'NaN/Inf in semantic token: {path}')
    return torch.from_numpy(np.ascontiguousarray(token)).float()


def build_model(checkpoint_path, device, args):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    network_opt = dict(config.get('network', {}))
    network_opt.pop('type', None)
    desired = {
        'use_semantic_codebook': bool(network_opt.get('use_semantic_codebook', True)),
        'use_dynamic_kernel': bool(network_opt.get('use_dynamic_kernel', True)),
        'use_frequency_experts': bool(network_opt.get('use_frequency_experts', True)),
    }
    network_opt['backbone_ckpt'] = None
    network_opt['residual_vq_ckpt'] = None
    network_opt['use_semantic_codebook'] = False
    model = GeoCodeSR(**network_opt)
    model.load_full_checkpoint(checkpoint_path, strict=True)
    model.use_semantic_codebook = desired['use_semantic_codebook']
    model.use_dynamic_kernel = desired['use_dynamic_kernel']
    model.use_frequency_experts = desired['use_frequency_experts']
    if args.baseline_only:
        model.use_semantic_codebook = False
        model.use_dynamic_kernel = False
        model.use_frequency_experts = False
    if args.disable_semantic_codebook:
        model.use_semantic_codebook = False
    if args.disable_dynamic_kernel:
        model.use_dynamic_kernel = False
    if args.disable_frequency_experts:
        model.use_frequency_experts = False
    model.to(device).eval()
    print('Enabled components: semantic_codebook={}, dynamic_reassembly={}, frequency_experts={}'.format(
        model.use_semantic_codebook, model.use_dynamic_kernel, model.use_frequency_experts))
    return model


def save_tif(path, image, metadata, output_dtype='source'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = metadata['profile'].copy()
    if output_dtype == 'source':
        dtype = metadata['dtype']
        data = image * metadata['input_range']
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            data = np.clip(np.rint(data), limits.min, limits.max)
        data = data.astype(dtype)
    else:
        dtype = np.dtype('float32')
        data = image.astype(dtype)
    profile.update(driver='GTiff', height=image.shape[0], width=image.shape[1],
                   count=image.shape[2], dtype=dtype.name)
    with rasterio.open(path, 'w', **profile) as destination:
        destination.write(np.transpose(data, (2, 0, 1)))
        if metadata['dataset_tags']:
            destination.update_tags(**metadata['dataset_tags'])
        for index, tags in enumerate(metadata['band_tags'], start=1):
            if tags:
                destination.update_tags(index, **tags)
        destination.descriptions = metadata['descriptions']
        destination.units = metadata['units']
        destination.colorinterp = metadata['colorinterp']
        destination.scales = metadata['scales']
        destination.offsets = metadata['offsets']


def metadata_from_lq(metadata, shape):
    output = dict(metadata)
    profile = metadata['profile'].copy()
    transform = profile.get('transform')
    if transform is not None:
        x_ratio = profile['width'] / shape[1]
        y_ratio = profile['height'] / shape[0]
        profile['transform'] = transform * Affine.scale(x_ratio, y_ratio)
    profile.update(height=shape[0], width=shape[1], count=shape[2])
    output['profile'] = profile
    return output


def psnr(sr, gt):
    rmse = float(np.sqrt(np.mean((sr - gt) ** 2)))
    return float('inf') if rmse < 1e-12 else 20.0 * math.log10(1.0 / rmse)


def sam(sr, gt):
    prediction = sr.reshape(-1, sr.shape[2]).astype(np.float64)
    target = gt.reshape(-1, gt.shape[2]).astype(np.float64)
    cosine = np.sum(prediction * target, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-8)
    return float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))) * 180.0 / math.pi)


def metrics(sr, gt):
    difference = sr - gt
    return {
        'mae': float(np.mean(np.abs(difference))),
        'rmse': float(np.sqrt(np.mean(difference ** 2))),
        'psnr': psnr(sr, gt),
        'sam': sam(sr, gt),
    }


def rgb_uint8(image):
    return (np.clip(image[:, :, :3], 0.0, 1.0) * 255.0).round().astype(np.uint8)


def minmax_uint8(image):
    minimum, maximum = float(image.min()), float(image.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    return (np.clip((image - minimum) / (maximum - minimum), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def save_preview(path, lq, sr, gt=None):
    lq_up = cv2.resize(lq, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)
    panels = [rgb_uint8(lq_up), rgb_uint8(sr)]
    if gt is not None:
        panels.extend([rgb_uint8(gt), minmax_uint8(np.abs(sr - gt)[:, :, :3])])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))


def run_batched_inference(args, model, device, lq_paths, output_dir):
    """Fast path for equally sized inference tiles; metadata remains per source TIFF."""
    pending = [path for path in lq_paths
               if not (args.resume and (output_dir / path.name).exists())]
    skipped = len(lq_paths) - len(pending)
    saved = 0
    for start in range(0, len(pending), args.batch_size):
        paths = pending[start:start + args.batch_size]
        images, metadata_items, semantics = [], [], []
        for path in paths:
            image, metadata = read_tif(path, args.bands, args.input_range)
            images.append(image)
            metadata_items.append(metadata)
            semantics.append(read_token(Path(args.dataroot_semantic) / f'{path.stem}.npy'))
        shapes = {image.shape for image in images}
        if len(shapes) != 1:
            raise ValueError(f'All TIFFs in a batch must have the same shape, got {shapes}. Use --batch_size 1.')
        lq_tensor = torch.stack([
            torch.from_numpy(image.transpose(2, 0, 1)) for image in images]).to(device)
        semantic_tensor = torch.stack(semantics).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                output = model(lq_tensor, semantic_tensor)
        batch_sr = output['sr'].float().cpu().numpy().transpose(0, 2, 3, 1)
        for path, image, sr, metadata in zip(paths, images, batch_sr, metadata_items):
            sr = np.clip(sr, 0.0, 1.0)
            expected_shape = (image.shape[0] * args.scale, image.shape[1] * args.scale, args.bands)
            if sr.shape != expected_shape:
                raise ValueError(f'Model output shape {sr.shape} does not match expected shape {expected_shape}.')
            save_tif(output_dir / path.name, sr, metadata_from_lq(metadata, sr.shape), args.output_dtype)
            saved += 1
        if saved % 100 < len(paths) or saved == len(pending):
            print(f'Saved {saved}/{len(pending)} new TIFFs (skipped existing: {skipped})')
    print(f'Done. saved={saved}, skipped={skipped}. Output dir: {output_dir}')


def main():
    args = parse_args()
    if args.bands != 3:
        raise ValueError('The current GeoCodeSR rebuild requires --bands 3.')
    checkpoint = Path(args.ckpt)
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint, device, args)
    lq_paths = list_tifs(args.dataroot_lq)
    output_dir = Path(args.output_dir)
    metric_rows = []
    skipped = 0
    print(f'Checkpoint: {checkpoint}\nDevice: {device}\nLR files: {len(lq_paths)}')
    if args.batch_size > 1:
        if args.save_preview:
            raise ValueError('--save_preview currently requires --batch_size 1.')
        run_batched_inference(args, model, device, lq_paths, output_dir)
        return
    for lq_path in lq_paths:
        try:
            output_path = output_dir / lq_path.name
            if args.resume and output_path.exists():
                skipped += 1
                continue
            lq, lq_metadata = read_tif(lq_path, args.bands, args.input_range)
            semantic = read_token(Path(args.dataroot_semantic) / f'{lq_path.stem}.npy').unsqueeze(0).to(device)
            lq_tensor = torch.from_numpy(lq.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                    output = model(lq_tensor, semantic)
            sr = np.clip(output['sr'].squeeze(0).cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)
            expected_shape = (lq.shape[0] * args.scale, lq.shape[1] * args.scale, args.bands)
            if sr.shape != expected_shape:
                raise ValueError(
                    f'Model output shape {sr.shape} does not match the requested x{args.scale} '
                    f'shape {expected_shape}. Use the scale matching the trained model.')
            gt, gt_metadata = None, None
            gt_path = Path(args.dataroot_gt) / lq_path.name
            if gt_path.exists():
                gt, gt_metadata = read_tif(gt_path, args.bands, args.input_range)
            save_metadata = gt_metadata if gt_metadata else metadata_from_lq(lq_metadata, sr.shape)
            save_tif(output_path, sr, save_metadata, args.output_dtype)
            if gt is not None and gt.shape == sr.shape:
                row = metrics(sr, gt)
                metric_rows.append(row)
                route_load = output['expert_load'].detach().cpu().numpy()
                print(f'{lq_path.name}: ' + ', '.join(f'{key}={value:.6f}' for key, value in row.items())
                      + f', route_load={route_load.tolist()}, route_entropy={float(output["routing_entropy"]):.6f}')
            if args.save_preview:
                save_preview(output_dir / 'preview' / f'{lq_path.stem}_preview.png', lq, sr, gt)
        except Exception as error:
            print(f'[WARN] Failed to process {lq_path}: {error}')
    if metric_rows:
        print('\nAverage metrics:')
        for key in metric_rows[0]:
            print(f'  {key.upper()}: {np.mean([row[key] for row in metric_rows]):.6f}')
    print(f'Done. skipped={skipped}. Output dir: {output_dir}')


if __name__ == '__main__':
    main()
